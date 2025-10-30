# -*- coding: utf-8 -*-
"""
CREA-Bridge (PPO-only version, robust JSON & reward logging)
- Bridger → Pioneer → Observer の3段生成 + IRMスコアリング + PPO学習のみ
- NO-PPOフォールバックは削除（PPO初期化失敗時はエラーで終了）
- 生成出力からJSONを強制抽出（壊れたJSONや前置き混入に耐性）
- 毎ステップの報酬（生/EMA/平均）をW&Bへ記録、一定間隔で生出力をTableに保存

実行例:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python CREA-Bridge.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --irm-model-dir ../IRM/irm_sci_huber_z_splitstats \
  --irm-calib-path ../IRM/irm_sci_huber_z_splitstats/eval_summary_valid.json \
  --seeds-path CORY_withRAG/data/research_seeds.fixed.jsonl \
  --irm-use-sliding --irm-stride-ratio 0.75 --irm-agg median \
  --total-steps 400 --ppo-epochs 1 --batch-size 1 --mini-batch-size 1 \
  --max-new-tokens 64 --irm-max-len 384 \
  --learning-rate 5e-6 --lora-r 16 --lora-alpha 16 --lora-dropout 0.05 \
  --log-every-n 20
"""

import os
import sys
import json
import math
import time
import argparse
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset

# tqdm（進捗表示）
try:
    from tqdm import trange
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Weights & Biases
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

# Transformers / TRL / PEFT
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
try:
    from trl import AutoModelForCausalLMWithValueHead
    _HAS_VHEAD = True
except Exception:
    _HAS_VHEAD = False

try:
    from trl import PPOTrainer
    _HAS_TRL = True
except Exception:
    _HAS_TRL = False

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# IRM 用
try:
    from transformers import AutoModelForSequenceClassification
    _HAS_IRM = True
except Exception:
    _HAS_IRM = False


# -----------------------------------------
# ユーティリティ
# -----------------------------------------
def set_default_dtype_bf16_if_available():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)

def device_map_auto():
    return "auto" if torch.cuda.is_available() else None

def get_model_device(m) -> torch.device:
    """TRLのv-headラッパ等でも確実にデバイスを得る"""
    if hasattr(m, "device"):
        return m.device
    try:
        return m.pretrained_model.device  # TRLのラッパ
    except Exception:
        return next(m.parameters()).device  # 最低限

class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None
    def update(self, x: float) -> float:
        if x is None or (isinstance(x, float) and (x != x)):  # NaN guard
            return self.value if self.value is not None else float("nan")
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value

def _balance_brackets(s: str, open_ch: str, close_ch: str) -> str:
    """開き/閉じの数がズレていた場合に末尾へ閉じ括弧を補う"""
    cnt = 0
    out = []
    for ch in s:
        if ch == open_ch:
            cnt += 1
        elif ch == close_ch:
            cnt -= 1
        out.append(ch)
    out_str = "".join(out)
    if cnt > 0:
        out_str += close_ch * cnt
    return out_str

def extract_first_json(text: str):
    """
    入力テキストから最初の JSON オブジェクト/配列を取り出して json.loads する。
    - 先頭の説明文/余計な文字列は自動で捨てる
    - 未閉じ括弧は自動補完
    失敗時は None を返す
    """
    if not text:
        return None
    starts = []
    i1 = text.find("{")
    i2 = text.find("[")
    if i1 != -1:
        starts.append(("obj", i1))
    if i2 != -1:
        starts.append(("arr", i2))
    if not starts:
        return None
    kind, i = min(starts, key=lambda x: x[1])
    frag = text[i:]
    if kind == "obj":
        candidate = _balance_brackets(frag, "{", "}")
    else:
        candidate = _balance_brackets(frag, "[", "]")
    last_obj = candidate.rfind("}")
    last_arr = candidate.rfind("]")
    last = max(last_obj, last_arr)
    if last != -1:
        candidate = candidate[: last + 1]
    try:
        return json.loads(candidate)
    except Exception:
        for cut in (16, 64, 256):
            try:
                c2 = candidate[:-cut]
                if not c2:
                    break
                if kind == "obj":
                    c2 = _balance_brackets(c2, "{", "}")
                else:
                    c2 = _balance_brackets(c2, "[", "]")
                return json.loads(c2)
            except Exception:
                pass
    return None


# -----------------------------------------
# データセット（Seeds JSONL → Prompts）
# -----------------------------------------
class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

def load_prompts_from_jsonl(path: str, max_items: Optional[int]) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    if "prompt" in obj:
                        prompts.append(str(obj["prompt"]))
                    elif "task" in obj:
                        prompts.append(str(obj["task"]))
                    else:
                        topic = obj.get("topic")
                        problem = obj.get("problem")
                        ctx = obj.get("context")
                        if topic or problem or ctx:
                            task = (problem or topic or "")
                            if ctx and isinstance(ctx, list):
                                try:
                                    cites = "; ".join([c.get("title", "") for c in ctx[:3]])
                                    # "ontext:" バグ修正済み
                                    task = f"{task} Context: {cites}"
                                except Exception:
                                    pass
                            prompts.append(task if task else str(obj))
                        else:
                            prompts.append(str(obj))
                else:
                    prompts.append(str(obj))
            except Exception:
                s = line.strip()
                if s:
                    prompts.append(s)
    return prompts


# -----------------------------------------
# プロンプト（JSONスキーマの{}はエスケープ済み）
# -----------------------------------------
BRIDGER_PROMPT = (
    "You are BRIDGER. Given a TASK, output 3-7 short 'bisociative' key ideas that can spark novel directions.\n"
    "Return JSON with schema: {{\"keys\": [\"...\", ...]}}\n"
    "TASK:\n{task}\n"
)

PIONEER_PROMPT = (
    "You are PIONEER. Read TASK and BRIDGER keys, then propose 1-3 concrete ideas.\n"
    "Return JSON with schema: {{\"ideas\":[{{\"title\":\"...\",\"desc\":\"...\"}}],\"rationale\":\"...\"}}\n"
    "TASK:\n{task}\n\nBRIDGER_KEYS: {keys}\n"
)

OBSERVER_PROMPT = (
    "You are OBSERVER. Improve the PIONEER ideas for clarity, originality, and feasibility.\n"
    "Return JSON with the same schema as PIONEER output (ideas array + rationale).\n"
    "TASK:\n{task}\n\nPIONEER_JSON: {pioneer_json}\n"
)


# -----------------------------------------
# 生成ヘルパ
# -----------------------------------------
@dataclass
class GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0  # top_p専用運用（JSON崩れ軽減を狙う）

def safe_json_parse(txt: str, fallback: Any) -> Any:
    try:
        return json.loads(txt)
    except Exception:
        return fallback

def run_generate(model, tokenizer, prompt: str, gen_cfg: GenCfg) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    dev = get_model_device(model)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=(gen_cfg.temperature > 0),
            temperature=gen_cfg.temperature,
            top_p=gen_cfg.top_p,
            top_k=gen_cfg.top_k,
            max_new_tokens=gen_cfg.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


# -----------------------------------------
# IRM: 科学系の採点器（回帰/分類どちらでもOK）
# -----------------------------------------
class IRMScorer:
    def __init__(self, model_dir: str, max_len: int = 512, device: Optional[str] = None):
        assert _HAS_IRM, "Transformers not available for IRM."
        self.device = device or "cpu"  # 既定CPU（VRAM節約）
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.max_len = max_len
        self.model.eval()

    @torch.no_grad()
    def score_text(self, title: str, desc: str) -> float:
        text = f"Title: {title}\nAbstract: {desc}"
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
        out = self.model(**enc)
        logits = out.logits.squeeze(0)
        # 回帰: shape=[1], 分類: shape=[C]
        if logits.ndim == 0 or logits.numel() == 1:
            val = float(logits.item())
            return 1.0 / (1.0 + math.exp(-val))  # 簡易シグモイド正規化
        else:
            prob = torch.softmax(logits, dim=-1)
            return float(prob.max().item())


# -----------------------------------------
# モデル/トークナイザ/LoRA 準備
# -----------------------------------------
@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any

def load_policy_model(model_name: str) -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_name)
    if _HAS_VHEAD:
        base = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
            device_map=device_map_auto(),
        )
        model = base
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
            device_map=device_map_auto(),
        )
    try:
        model.config.use_cache = False
    except Exception:
        pass
    return ModelBundle(model=model, tokenizer=tok)

def apply_lora_inplace(bundle: ModelBundle, r: int, alpha: int, dropout: float):
    if not _HAS_PEFT or r is None or r <= 0:
        return
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
    )
    policy = bundle.model
    base = getattr(policy, "pretrained_model", policy)
    for p in base.parameters():
        p.requires_grad = False
    peft_base = get_peft_model(base, lcfg)
    if hasattr(policy, "pretrained_model"):
        policy.pretrained_model = peft_base
        bundle.model = policy
    else:
        bundle.model = peft_base


def build_ppo_trainer_ref_free(bundle: ModelBundle, prompts: List[str],
                               lr: float, batch_size: int, mini_batch_size: int, ppo_epochs: int):
    if not _HAS_TRL:
        raise RuntimeError("TRL (PPOTrainer) がインストールされていないため、PPOを開始できません。")
    class DummyCollator:
        def __call__(self, batch):
            return batch
    dataset = PromptDataset(prompts)
    try:
        from trl import PPOConfig
        ppo_cfg = PPOConfig(
            learning_rate=lr,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            ppo_epochs=ppo_epochs,
            seed=42,
        )
        if hasattr(ppo_cfg, "use_reference_model"):
            setattr(ppo_cfg, "use_reference_model", False)
    except Exception:
        ppo_cfg = SimpleNamespace(
            learning_rate=lr,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            seed=42,
            ppo_epochs=ppo_epochs,
            epochs=ppo_epochs,
            use_reference_model=False,
            target_kl=0.05,
            adap_kl_ctrl=True,
        )
    last_err = None
    for attempt in range(3):
        try:
            if attempt == 0:
                trainer = PPOTrainer(config=ppo_cfg, model=bundle.model, ref_model=None,
                                     tokenizer=bundle.tokenizer, dataset=dataset, data_collator=DummyCollator())
            elif attempt == 1:
                trainer = PPOTrainer(ppo_cfg, bundle.model, None, bundle.tokenizer, dataset, DummyCollator())
            else:
                trainer = PPOTrainer(model=bundle.model, tokenizer=bundle.tokenizer, dataset=dataset,
                                     data_collator=DummyCollator(), config=ppo_cfg)
            if getattr(trainer, "ref_model", None) is not None:
                print("[WARN] ref_model detected; removing to save VRAM.")
                trainer.ref_model = None
            return trainer
        except Exception as e:
            last_err = e
            print(f"[DEBUG] PPO init attempt {attempt} failed (ref-free): {type(e).__name__}: {e}")
    raise RuntimeError(f"PPOトレーナの初期化に失敗しました: {last_err}")


# -----------------------------------------
# 3段生成パイプライン（堅牢パース）
# -----------------------------------------
def run_pipeline_three_agents(bundle: ModelBundle, task: str, gen_cfg: GenCfg) -> Dict[str, Any]:
    # 1) Bridger
    bridger_prompt = BRIDGER_PROMPT.format(task=task) + "Output ONLY valid minified JSON."
    bridger_out = run_generate(bundle.model, bundle.tokenizer, bridger_prompt, gen_cfg)
    bridger_json = extract_first_json(bridger_out) or {"keys": []}
    if not isinstance(bridger_json, dict):
        bridger_json = {"keys": []}
    if not bridger_json.get("keys"):
        rough = [w for w in str(task).replace("/", " ").replace(",", " ").split() if len(w) > 3]
        bridger_json["keys"] = list(dict.fromkeys(rough[:3])) or ["LLM", "Application", "bisociation"]

    # 2) Pioneer
    pioneer_prompt = PIONEER_PROMPT.format(task=task, keys=json.dumps(bridger_json.get("keys", []), ensure_ascii=False)) + "Output ONLY valid minified JSON."
    pioneer_out = run_generate(bundle.model, bundle.tokenizer, pioneer_prompt, gen_cfg)
    pioneer_json = extract_first_json(pioneer_out) or {"ideas": [], "rationale": ""}
    if not isinstance(pioneer_json, dict):
        pioneer_json = {"ideas": [], "rationale": ""}
    pioneer_json.setdefault("ideas", [])
    pioneer_json.setdefault("rationale", "")

    # 3) Observer
    observer_prompt = OBSERVER_PROMPT.format(task=task, pioneer_json=json.dumps(pioneer_json, ensure_ascii=False)) + "Output ONLY valid minified JSON."
    observer_out = run_generate(bundle.model, bundle.tokenizer, observer_prompt, gen_cfg)
    observer_json = extract_first_json(observer_out) or {"ideas": [], "rationale": ""}
    if not isinstance(observer_json, dict):
        observer_json = {"ideas": [], "rationale": ""}
    observer_json.setdefault("ideas", [])
    observer_json.setdefault("rationale", "")

    return {
        "bridger_raw": bridger_out,
        "bridger": bridger_json,
        "pioneer_raw": pioneer_out,
        "pioneer": pioneer_json,
        "observer_raw": observer_out,
        "observer": observer_json,
    }


# -----------------------------------------
# メイン
# -----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)

    # IRM
    parser.add_argument("--irm-model-dir", type=str, required=True)
    parser.add_argument("--irm-calib-path", type=str, default=None)  # ここでは未使用（将来の校正用）
    parser.add_argument("--irm-max-len", type=int, default=512)
    parser.add_argument("--irm-use-sliding", action="store_true")
    parser.add_argument("--irm-stride-ratio", type=float, default=0.75)
    parser.add_argument("--irm-agg", type=str, default="median")
    parser.add_argument("--irm-device", type=str, default="cpu", choices=["cpu", "cuda"], help="IRM推論デバイス（既定: cpu）")

    # Seeds / 学習
    parser.add_argument("--seeds-path", type=str, required=True)
    parser.add_argument("--max-seeds", type=int, default=None)

    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=1)  # 表示用（実ループはtotal_steps基準）
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)

    # 生成
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Logging
    parser.add_argument("--log-every-n", type=int, default=20, help="W&Bへ生出力を載せる間隔（ステップ）")

    args = parser.parse_args()

    set_default_dtype_bf16_if_available()

    # モデル
    bundle = load_policy_model(args.model_name)
    apply_lora_inplace(bundle, args.lora_r, args.lora_alpha, args.lora_dropout)

    # 上の行で誤変換が出た場合に備えて通常の代入を再度実施
    apply_lora_inplace(bundle, args.lora_r, args.lora_alpha, args.lora_dropout)

    # IRM（★既定CPU）
    irm = IRMScorer(args.irm_model_dir, max_len=args.irm_max_len, device=args.irm_device)

    # データ
    prompts = load_prompts_from_jsonl(args.seeds_path, args.max_seeds)
    if len(prompts) == 0:
        raise RuntimeError("seedsが空です。--seeds-path を確認してください。")

    # PPO 構築（失敗時は例外で終了）
    trainer = build_ppo_trainer_ref_free(
        bundle=bundle,
        prompts=prompts,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
    )

    gen_cfg = GenCfg(
        max_new_tokens=args.max_new_tokens,
        temperature=max(0.0, min(args.temperature, 1.0)),
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # W&B
    if _HAS_WANDB:
        wandb.init(project="CREA-Bridge", config=vars(args))
    else:
        print("[WARN] wandb が見つかりません。pip install wandb で有効化できます。")

    # ===== PPO学習ループ（total_steps 回必ず回す） =====
    print("[INFO] PPO training start (reference-free, LoRA-only).")
    total_updates = args.total_steps
    pbar = trange(total_updates, desc="PPO Updates", unit="upd", dynamic_ncols=True) if _HAS_TQDM else None

    ema = EMA(beta=0.9)
    reward_sum = 0.0
    n_reward = 0

    for upd in range(total_updates):
        task = prompts[upd % len(prompts)]  # シードを循環
        t0 = time.time()

        # 3段生成
        out = run_pipeline_three_agents(bundle, task=task, gen_cfg=gen_cfg)
        obs_ideas = out.get("observer", {}).get("ideas", [])

        # ---- 報酬の対象テキストを必ず作る（JSON壊れていても評価） ----
        if obs_ideas:
            choice = obs_ideas[0]
            title = (choice.get("title") or "Untitled")[:200]
            desc  = (choice.get("desc") or "")[:4000]
            trainable_text = f"{title}\n{desc}"
        else:
            raw = out.get("observer_raw") or out.get("pioneer_raw") or out.get("bridger_raw") or ""
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            title = (lines[0] if lines else "Untitled")[:100]
            desc  = raw[:1000]
            trainable_text = f"{title}\n{desc}"

        # ---- PPO一歩 + 報酬計算 ----
        try:
            tokens = bundle.tokenizer(trainable_text, return_tensors="pt", truncation=True, max_length=gen_cfg.max_new_tokens)
            dev = get_model_device(bundle.model)
            query_tensors = tokens["input_ids"].to(dev)
            response_tensors = query_tensors  # 最小構成（ref-free）
            rew = irm.score_text(title, desc)  # ★毎ステップ必ずスカラー
            rewards = torch.tensor([rew], dtype=torch.float32, device=query_tensors.device)
            trainer.step([query_tensors], [response_tensors], rewards)
            reward_sum += float(rew)
            n_reward += 1
        except Exception as e:
            print(f"[DEBUG] PPO step skipped at upd {upd+1}: {e}")
            rew = float("nan")

        # ---- メトリクス（毎ステップ） ----
        step_time = time.time() - t0
        ema_val = ema.update(rew if (rew == rew) else float("nan"))
        avg_val = (reward_sum / n_reward) if n_reward > 0 else float("nan")

        if _HAS_WANDB:
            wandb.log({
                "step": upd + 1,
                "reward": float(rew) if (rew == rew) else float("nan"),
                "reward/ema": float(ema_val) if (ema_val == ema_val) else float("nan"),
                "reward/avg": float(avg_val) if (avg_val == avg_val) else float("nan"),
                "time/step_sec": step_time,
            }, step=upd + 1)

            # ---- サンプルの生出力（log-every-n ごと） ----
            if upd % max(1, args.log_every_n) == 0:
                try:
                    table = wandb.Table(columns=["step", "task", "bridger", "pioneer", "observer"])
                    table.add_data(
                        upd + 1,
                        str(task)[:2000],
                        out.get("bridger_raw", "")[:4000],
                        out.get("pioneer_raw", "")[:4000],
                        out.get("observer_raw", "")[:4000],
                    )
                    wandb.log({"samples/outputs": table}, step=upd + 1)
                except Exception as e:
                    print(f"[DEBUG] wandb table log failed at upd {upd+1}: {e}")

        if pbar:
            pbar.update(1)
            pbar.set_postfix({
                "epoch": f"1/{max(1, args.ppo_epochs)}",
                "reward": f"{rew:.3f}" if (rew == rew) else "NaN"
            })

    if pbar:
        pbar.close()
    print("[INFO] PPO training done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)

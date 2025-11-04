# -*- coding: utf-8 -*-
"""
CREA-Bridge (Dual-GPU: policy=CUDA:X, IRM=CUDA:Y)
- Bridger → Pioneer → Observer の3段生成 + IRMスコア + PPO更新
- policy は --policy-device（既定: cuda:0）
- IRM は --irm-device（既定: cuda:1）
- 生成は trainer.generate を使わず、policy.generate を直叩き（policy側GPU）
- "実質 reference-free": ref_model は model を渡すが KL=0 で参照影響をゼロ化
- TRL が rewards を「スカラーTensorのリスト」または「トークン長ベクトルのリスト」で期待する版を想定
"""

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# PPOに渡すクエリ/応答の最大トークン長（VRAM節約用）
MAX_QUERY_TOKENS_FOR_PPO = 256
MAX_RESP_TOKENS_FOR_PPO  = 128

import torch
from torch.utils.data import Dataset

# tqdm / wandb（任意）
try:
    from tqdm import trange
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model


# ---------------- ユーティリティ ----------------
def assert_cuda_device(dev_str: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA が利用できません。GPU 環境で実行してください。")
    d = torch.device(dev_str)
    if d.type != "cuda":
        raise RuntimeError(f"{dev_str} は CUDA ではありません。'cuda:0' / 'cuda:1' などを指定してください。")
    if d.index is None:
        raise RuntimeError(f"{dev_str} の index が取得できません。'cuda:0' / 'cuda:1' を明示してください。")
    return d

def set_default_dtype_bf16_if_available():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)

def _preferred_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None
    def update(self, x: float) -> float:
        if x is None or (isinstance(x, float) and (x != x)):  # NaN
            return self.value if self.value is not None else float("nan")
        self.value = x if self.value is None else (self.beta*self.value + (1-self.beta)*x)
        return self.value

def _balance_brackets(s: str, open_ch: str, close_ch: str) -> str:
    cnt = 0
    out = []
    for ch in s:
        if ch == open_ch: cnt += 1
        elif ch == close_ch: cnt -= 1
        out.append(ch)
    out_str = "".join(out)
    if cnt > 0:
        out_str += close_ch * cnt
    return out_str

def extract_first_json(text: str):
    if not text:
        return None
    i1, i2 = text.find("{"), text.find("[")
    starts = []
    if i1 != -1: starts.append(("obj", i1))
    if i2 != -1: starts.append(("arr", i2))
    if not starts:
        return None
    kind, i = min(starts, key=lambda x: x[1])
    frag = text[i:]
    candidate = _balance_brackets(frag, "{" if kind=="obj" else "[", "}" if kind=="obj" else "]")
    last = max(candidate.rfind("}"), candidate.rfind("]"))
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
                c2 = _balance_brackets(c2, "{" if kind=="obj" else "[", "}" if kind=="obj" else "]")
                return json.loads(c2)
            except Exception:
                pass
    return None


# ---------------- データセット ----------------
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


# ---------------- プロンプト（{}はエスケープ済） ----------------
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


# ---------------- 生成（policy_device固定） ----------------
@dataclass
class GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0

def run_generate_cuda(model, tokenizer, prompt: str, gen_cfg: GenCfg, policy_device: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(policy_device) for k, v in enc.items()}
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


# ---------------- IRM（irm_device固定） ----------------
class IRMScorer:
    def __init__(self, model_dir: str, max_len: int = 512, device: str = "cuda:1"):
        dev = assert_cuda_device(device)  # CPU フォールバック禁止
        self.device = str(dev)
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_len = max_len

    @torch.no_grad()
    def score_text(self, title: str, desc: str) -> float:
        text = f"Title: {title}\nAbstract: {desc}"
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits.squeeze(0)
        if logits.ndim == 0 or logits.numel() == 1:
            val = float(logits.item())
            return 1.0 / (1.0 + math.exp(-val))
        else:
            prob = torch.softmax(logits, dim=-1)
            return float(prob.max().item())


# ---------------- モデル / LoRA（policy_device固定） ----------------
@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any

def load_policy_model(model_name: str, policy_device: str) -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = _preferred_dtype()

    base = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None,   # 明示的に後で to(policy_device)
    )
    base.to(policy_device)

    if not hasattr(base, "v_head") or base.v_head is None:
        from trl.models.modeling_value_head import ValueHead
        hs = base.pretrained_model.config.hidden_size if hasattr(base, "pretrained_model") else base.config.hidden_size
        base.v_head = ValueHead(hs).to(policy_device)
    try:
        base.config.use_cache = False
    except Exception:
        pass
    return ModelBundle(model=base, tokenizer=tok)

def apply_lora_inplace(bundle: ModelBundle, r: int, alpha: int, dropout: float, policy_device: str):
    if r is None or r <= 0:
        return
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules
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
    bundle.model.to(policy_device)
    if hasattr(bundle.model, "v_head") and bundle.model.v_head is not None:
        bundle.model.v_head.to(policy_device)


# ---------------- 3-Agent パイプライン ----------------
def run_pipeline_three_agents(bundle: ModelBundle, task: str, gen_cfg: GenCfg, policy_device: str) -> Dict[str, Any]:
    bridger_out = run_generate_cuda(
        bundle.model, bundle.tokenizer,
        BRIDGER_PROMPT.format(task=task) + " Output ONLY valid minified JSON.",
        gen_cfg, policy_device
    )
    bridger_json = extract_first_json(bridger_out) or {"keys": []}
    if not isinstance(bridger_json, dict):
        bridger_json = {"keys": []}
    if not bridger_json.get("keys"):
        rough = [w for w in str(task).replace("/", " ").replace(",", " ").split() if len(w) > 3]
        bridger_json["keys"] = list(dict.fromkeys(rough[:3])) or ["LLM", "Application", "bisociation"]

    pioneer_out = run_generate_cuda(
        bundle.model, bundle.tokenizer,
        PIONEER_PROMPT.format(task=task, keys=json.dumps(bridger_json.get("keys", []), ensure_ascii=False))
        + " Output ONLY valid minified JSON.",
        gen_cfg, policy_device
    )
    pioneer_json = extract_first_json(pioneer_out) or {"ideas": [], "rationale": ""}
    if not isinstance(pioneer_json, dict):
        pioneer_json = {"ideas": [], "rationale": ""}
    pioneer_json.setdefault("ideas", [])
    pioneer_json.setdefault("rationale", "")

    observer_out = run_generate_cuda(
        bundle.model, bundle.tokenizer,
        OBSERVER_PROMPT.format(task=task, pioneer_json=json.dumps(pioneer_json, ensure_ascii=False))
        + " Output ONLY valid minified JSON.",
        gen_cfg, policy_device
    )
    observer_json = extract_first_json(observer_out) or {"ideas": [], "rationale": ""}
    if not isinstance(observer_json, dict):
        observer_json = {"ideas": [], "rationale": ""}
    observer_json.setdefault("ideas", [])
    observer_json.setdefault("rationale", "")

    return {
        "bridger_raw": bridger_out, "bridger": bridger_json,
        "pioneer_raw": pioneer_out, "pioneer": pioneer_json,
        "observer_raw": observer_out, "observer": observer_json,
    }


# ---------------- メイン ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, required=True)

    # IRM
    ap.add_argument("--irm-model-dir", type=str, required=True)
    ap.add_argument("--irm-calib-path", type=str, default=None)
    ap.add_argument("--irm-max-len", type=int, default=512)
    ap.add_argument("--irm-use-sliding", action="store_true")
    ap.add_argument("--irm-stride-ratio", type=float, default=0.75)
    ap.add_argument("--irm-agg", type=str, default="median")
    ap.add_argument("--irm-device", type=str, default="cuda:1")

    # Seeds
    ap.add_argument("--seeds-path", type=str, required=True)
    ap.add_argument("--max-seeds", type=int, default=None)

    # PPO / Train
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--mini-batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=5e-6)

    # Generation
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Logging
    ap.add_argument("--log-every-n", type=int, default=20)

    # Devices
    ap.add_argument("--policy-device", type=str, default="cuda:0")

    args = ap.parse_args()

    # CUDA 環境チェック
    assert_cuda_device(args.policy_device)
    assert_cuda_device(args.irm_device)
    if args.policy_device == args.irm_device:
        raise RuntimeError(f"--policy-device と --irm-device が同じです（{args.policy_device}）。別GPUを指定してください。")

    # 速度最適化（Ada世代）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 断片化対策
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_default_dtype_bf16_if_available()

    # Policy
    bundle = load_policy_model(args.model_name, args.policy_device)
    apply_lora_inplace(bundle, args.lora_r, args.lora_alpha, args.lora_dropout, args.policy_device)

    print("[SANITY] policy device ->", next(bundle.model.parameters()).device)
    if str(next(bundle.model.parameters()).device) != args.policy_device:
        raise RuntimeError(f"Policy model が {args.policy_device} にいません。コードを確認してください。")

    if hasattr(bundle.model, "v_head") and bundle.model.v_head is not None:
        bundle.model.v_head.to(args.policy_device)

    # IRM
    irm = IRMScorer(args.irm_model_dir, max_len=args.irm_max_len, device=args.irm_device)

    # Prompts
    prompts = load_prompts_from_jsonl(args.seeds_path, args.max_seeds)
    if not prompts:
        raise RuntimeError("seedsが空です。--seeds-path を確認してください。")

    # PPO（実質 reference-free: KL=0）
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        seed=42,
        init_kl_coef=0.0,
        target_kl=None,
        use_score_scaling=False,
        use_score_norm=False,
    )
    if hasattr(ppo_cfg, "use_reference_model"):
        setattr(ppo_cfg, "use_reference_model", False)

    dataset = PromptDataset(prompts)
    class DummyCollator:
        def __call__(self, batch): return batch

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=bundle.model,       # policy_device
        ref_model=bundle.model,   # ダミー参照（KL=0なので影響なし）
        tokenizer=bundle.tokenizer,
        dataset=dataset,
        data_collator=DummyCollator(),
    )

    trainer.model.to(args.policy_device).eval()
    if getattr(trainer, "ref_model", None) is not None:
        trainer.ref_model.to(args.policy_device).eval()

    gen_cfg = GenCfg(
        max_new_tokens=args.max_new_tokens,
        temperature=max(0.0, min(args.temperature, 1.0)),
        top_p=args.top_p,
        top_k=args.top_k,
    )

    if _HAS_WANDB:
        wandb.init(project="CREA-Bridge", config=vars(args))

    print(f"[INFO] PPO training start (policy={args.policy_device}, IRM={args.irm_device}, KL=0).")
    total_updates = args.total_steps
    pbar = trange(total_updates, desc="PPO Updates", unit="upd", dynamic_ncols=True) if _HAS_TQDM else None

    ema = EMA(0.9)
    reward_sum, n_reward = 0.0, 0

    for upd in range(total_updates):
        task = prompts[upd % len(prompts)]
        t0 = time.time()

        # ----- 3段生成（policy_device） -----
        out = run_pipeline_three_agents(bundle, task, gen_cfg, args.policy_device)
        obs_ideas = out.get("observer", {}).get("ideas", [])

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

        try:
            # 1) query: policy_device
            enc = bundle.tokenizer(
                trainable_text,
                return_tensors="pt",
                truncation=True,
                max_length=max(128, gen_cfg.max_new_tokens),
            )
            enc = {k: v.to(args.policy_device) for k, v in enc.items()}
            query_len = enc["input_ids"].shape[1]

            # 2) response: policy.generate（policy_device）
            with torch.no_grad():
                gen_out = bundle.model.generate(
                    **enc,
                    do_sample=(gen_cfg.temperature > 0),
                    temperature=gen_cfg.temperature,
                    top_p=gen_cfg.top_p,
                    top_k=gen_cfg.top_k,
                    max_new_tokens=gen_cfg.max_new_tokens,
                    pad_token_id=bundle.tokenizer.eos_token_id,
                    eos_token_id=bundle.tokenizer.eos_token_id,
                )
            resp_ids = gen_out[0, query_len:]
            if resp_ids.numel() == 0:
                resp_ids = gen_out[0, -1:].detach()

            # PPOに入れるトークンを上限でカット（VRAM節約）
            q_ids = enc["input_ids"][0]  # policy_device
            if q_ids.shape[0] > MAX_QUERY_TOKENS_FOR_PPO:
                q_ids = q_ids[-MAX_QUERY_TOKENS_FOR_PPO:]
            if resp_ids.shape[0] > MAX_RESP_TOKENS_FOR_PPO:
                resp_ids = resp_ids[:MAX_RESP_TOKENS_FOR_PPO]

            query_tensors    = [q_ids]        # policy_device
            response_tensors = [resp_ids]     # policy_device

            # 3) 報酬: IRM（irm_device）→ PPOへは「スカラーTensorのリスト」（policy_device上）
            rew = irm.score_text(title, desc)
            rewards = [torch.tensor(float(rew), dtype=torch.float32, device=args.policy_device)]
            # デバッグ表示
            print(f"[DBG] reward scalar -> device={rewards[0].device}, val={rewards[0].item():.6f}")

            # 4) PPO更新（policy_device）
            trainer.step(query_tensors, response_tensors, rewards)

            reward_sum += float(rew)
            n_reward += 1

        except Exception as e:
            print(f"[DEBUG] PPO step failed at upd {upd+1}: {repr(e)}")
            import traceback; traceback.print_exc()
            rew = float("nan")

        # ----- ログ -----
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
        # 2GPUが見えているかの最低限チェック（可視デバイス数）
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if vis is not None:
            if len([x for x in vis.split(",") if x.strip() != ""]) < 2:
                raise RuntimeError("CUDA_VISIBLE_DEVICES に少なくとも2つのGPUを指定してください（例: '0,1'）。")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("2枚以上のGPUが見えていません。nvidia-smi と CUDA_VISIBLE_DEVICES を確認してください。")

        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)
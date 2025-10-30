# trainer_builder.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
)
import random

# --- datasets が無い環境でも動くようにフォールバック ---
HAS_HF_DATASETS = True
try:
    from datasets import Dataset as HFDataset
except Exception:
    HAS_HF_DATASETS = False
    from torch.utils.data import Dataset as TorchDataset

    class PromptListDataset(TorchDataset):
        """datasets が無い場合の簡易フォールバック"""
        def __init__(self, prompts: List[str]):
            self.prompts = list(prompts)
        def __len__(self):
            return len(self.prompts)
        def __getitem__(self, idx):
            # TRL 側は dict で "prompt" キーを読む実装に合わせる
            return {"prompt": self.prompts[idx]}


# -----------------------------
# ユーティリティ
# -----------------------------
def _record_to_prompt(rec: Dict[str, Any]) -> Optional[str]:
    """
    JSONLの1レコード -> 文字列プロンプトへ変換。
    - 'prompt' があれば最優先
    - なければ topic/problem/sources/context から合成
    """
    if not isinstance(rec, dict):
        return None

    if "prompt" in rec and isinstance(rec["prompt"], str) and rec["prompt"].strip():
        return rec["prompt"].strip()

    topic = rec.get("topic")
    problem = rec.get("problem")
    constraints = rec.get("constraints")
    sources = rec.get("sources")
    ctx_list = rec.get("context")

    # sources をテキスト化
    if isinstance(sources, list):
        src_txt = ", ".join(map(str, sources))
    elif isinstance(sources, str):
        src_txt = sources
    else:
        src_txt = ""

    # context を数件まとめる
    ctx_lines: List[str] = []
    if isinstance(ctx_list, list):
        for c in ctx_list[:5]:
            if isinstance(c, dict):
                t = c.get("title", "")
                a = c.get("abstract", "")
                fields = c.get("fields", [])
                fields_str = ", ".join(fields) if isinstance(fields, list) else str(fields)
                ctx_lines.append(f"- {t} | fields: {fields_str} | abs: {a}")
            else:
                ctx_lines.append(f"- {str(c)}")

    headline = problem or topic or "Creative research idea generation"

    base = [
        f"Topic: {topic}" if topic else None,
        f"Problem: {problem}" if problem else None,
        f"Sources/Fields: {src_txt}" if src_txt else None,
        f"Constraints: {constraints}" if constraints else None,
    ]
    base = [b for b in base if b]

    ctx_block = "Context papers:\n" + "\n".join(ctx_lines[:10]) if ctx_lines else ""

    prompt = (
        f"{headline}\n"
        + ("\n".join(base) + "\n" if base else "")
        + (ctx_block + "\n\n" if ctx_block else "")
        + "Give me three concrete, novel, and feasible research ideas that bridge distant fields above. "
          "Each idea must include: (1) one-sentence title, (2) 2–3 sentences rationale explaining the bisociation, "
          "(3) minimal viable experiment with measurable metrics, (4) realistic risks and mitigations. "
          "Avoid vague claims; keep it implementable."
    )
    return prompt


def _load_prompts_from_jsonl(path: str, max_seeds: Optional[int]) -> List[str]:
    prompts: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # 1行そのままをプロンプトとして使う逃げ道
                    prompts.append(line)
                    continue
                p = _record_to_prompt(rec)
                if p:
                    prompts.append(p)
    except FileNotFoundError:
        print(f"[trainer_builder] WARNING: seeds file not found: {path}")

    random.shuffle(prompts)
    if max_seeds is not None and max_seeds > 0:
        prompts = prompts[:max_seeds]
    return prompts


def _ensure_non_empty_dataset(prompts: List[str]) -> HFDataset:
    """プロンプト0件でも必ず1件入れて Dataset を返す"""
    if len(prompts) == 0:
        print("[trainer_builder] WARNING: prompts loaded = 0. Fallback prompt will be used.")
        prompts = [
            "Give me three creative research ideas that connect two distant fields (LLMs × Reinforcement Learning). "
            "Provide title, rationale, minimal viable experiment, and risks with mitigations."
        ]

    print(f"[trainer_builder] prompts loaded: {len(prompts)}; head: {prompts[0][:120]}")
    dataset = HFDataset.from_dict({"prompt": prompts})

    if len(dataset) == 0:
        print("[trainer_builder] CRITICAL: dataset is still empty after fallback. Injecting one fallback prompt.")
        dataset = HFDataset.from_dict({
            "prompt": [
                "Give me three creative research ideas that connect two distant fields (LLMs × Reinforcement Learning). "
                "Provide title, rationale, minimal viable experiment, and risks with mitigations."
            ]
        })
    return dataset


def _get_kw_or_args(kwargs: Dict[str, Any], name: str, default: Any):
    """kwargs から取得。無ければ kwargs['args']（オブジェクト）から getattr で取得。"""
    if name in kwargs:
        return kwargs[name]
    args = kwargs.get("args", None)
    if args is not None:
        return getattr(args, name, default)
    return default

def _make_trainer(ppo_config, model, tokenizer, dataset):
    # 1) 新しめ: PPOTrainer(config=..., ...)
    try:
        return PPOTrainer(
            config=ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=None,
            ref_model=None  # あれば使われ、無ければ無視される
        )
    except TypeError:
        pass

    # 2) 中間: 先頭の位置引数に PPOConfig を渡す
    try:
        return PPOTrainer(
            ppo_config,
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=None,
            ref_model=None
        )
    except TypeError:
        pass

    # 3) 旧 API: config を受け付けない版（最小引数のみ）
    #    （この場合は PPOConfig の中身は trainer 側のデフォルトに従う）
    return PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=None,
    )



# -----------------------------
# メイン：PPOトレーナ構築
# -----------------------------
def build_ppo_trainer(model: torch.nn.Module, tokenizer, **kwargs) -> PPOTrainer:
    """
    TRL 0.24.0 向けのシンプル版。
    - 参照モデル（ref_model）は作らない → メモリ節約 & deepcopy回避
    - PPOConfig には互換性の高い最小限の引数のみ渡す
    - ppo_epochs 等は存在確認してから代入
    """
    # ---- seeds 読み込み ------------------------------------------------------
    seeds_path = _get_kw_or_args(kwargs, "seeds_path", None)
    max_seeds  = _get_kw_or_args(kwargs, "max_seeds", None)
    print(f"[trainer_builder] seeds_path={seeds_path}, max_seeds={max_seeds}")

    prompts: List[str] = []
    if seeds_path:
        prompts = _load_prompts_from_jsonl(seeds_path, max_seeds)
    dataset = _ensure_non_empty_dataset(prompts)

    # ---- LoRA 構成（必要に応じて）-------------------------------------------
    lora_r = int(_get_kw_or_args(kwargs, "lora_r", 0))
    if lora_r > 0:
        lora_alpha   = int(_get_kw_or_args(kwargs, "lora_alpha", 32))
        lora_dropout = float(_get_kw_or_args(kwargs, "lora_dropout", 0.05))
        lora_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets,
        )
        print(f"[trainer_builder] Applying LoRA to policy only: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        policy = model.pretrained_model            # AutoModelForCausalLM
        policy = get_peft_model(policy, lora_cfg)  # LoRA 適用
        model.pretrained_model = policy
        try:
            model.pretrained_model.config.use_cache = False
        except Exception:
            pass
    else:
        print("[trainer_builder] LoRA disabled (r=0).")

    # ---- PPOConfig（最小限） -------------------------------------------------
    learning_rate   = float(_get_kw_or_args(kwargs, "learning_rate", 1e-5))
    batch_size      = int(_get_kw_or_args(kwargs, "batch_size", 1))
    mini_batch_size = int(_get_kw_or_args(kwargs, "mini_batch_size", 1))
    ppo_epochs_val  = int(_get_kw_or_args(kwargs, "ppo_epochs", 4))

    # TRL 0.24.0 では constructor に余計な引数を渡さない
    ppo_config = PPOConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
    )

    # 参照モデルを使わない（存在確認してから安全に設定）
    if hasattr(ppo_config, "use_reference_model"):
        ppo_config.use_reference_model = False

    # 反復回数キーはバージョン差があるため存在チェックして設定
    if hasattr(ppo_config, "ppo_epochs"):
        ppo_config.ppo_epochs = ppo_epochs_val
    elif hasattr(ppo_config, "epochs"):
        ppo_config.epochs = ppo_epochs_val
    elif hasattr(ppo_config, "num_train_epochs"):
        ppo_config.num_train_epochs = ppo_epochs_val
    else:
        print("[trainer_builder] NOTE: This TRL PPOConfig has no epochs field; the default will be used.")

    # ---- PPOTrainer（TRL 0.24.0 形式）---------------------------------------
    # ref_model=None を明示し、tokenizer/dataset/data_collator を素直に渡す
    try:
        trainer = PPOTrainer(
            config=ppo_config,
            model=model,
            ref_model=None,          # ← 参照モデルを作らない
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=None,      # 'prompt' カラムをそのまま使う
        )
        return trainer
    except TypeError as e:
        # もし何かの理由でシグネチャが違っても最低限のフォールバック
        print(f"[trainer_builder] Fallback due to TypeError: {e}")
        trainer = PPOTrainer(
            ppo_config,              # 古いTRL互換（位置引数）
            model,
            None,                    # ref_model
            tokenizer,
            dataset,
            None,                    # data_collator
        )
        return trainer

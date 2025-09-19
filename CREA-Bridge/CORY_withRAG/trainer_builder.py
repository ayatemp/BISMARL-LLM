# -*- coding: utf-8 -*-
"""
trainer_builder.py (minimal, LoRA auto-detect, TRL PPO)
- TRL の PPOTrainer が要求する ValueHead 付きモデルを用意
- LoRA は ValueHead ラッパの内側 (pretrained_model) に適用
- モデルに合わせて LoRA の target_modules を自動推定（GPT-2 / LLaMA / Mistral / Falcon 等）
- 生成・保存の薄いラッパ _GenWrapper を返す（.generate / .step / .save_pretrained / .accelerator）
"""

from typing import Any, Dict, Optional, List

import torch
from transformers import AutoTokenizer

from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)


# -----------------------------
# LoRA 対象層の自動推定
# -----------------------------
def _guess_lora_targets(base_model) -> List[str]:
    """モデルのモジュール名から LoRA 対象層を推定"""
    names = [n for n, _ in base_model.named_modules()]

    def has(substr: str) -> bool:
        return any(substr in n for n in names)

    # LLaMA / Mistral / Qwen 系
    if has("q_proj"):
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # GPT-2 系
    if has("c_attn"):
        return ["c_attn", "c_proj", "c_fc"]

    # Falcon 系
    if has("query_key_value"):
        # 代表的な線形層名を候補に
        cand = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", "o_proj"]
        return [x for x in cand if has(x)]

    # 最終保険：名前に 'proj' を含む終端名をいくつか拾う
    fallback = sorted({n.split(".")[-1] for n in names if "proj" in n})[:8]
    return fallback or ["q_proj", "v_proj", "o_proj"]


def _maybe_apply_lora_to_inner(model_with_vhead, lora: Optional[Dict[str, Any]]) -> bool:
    """
    ValueHead ラッパの内側 (pretrained_model) に LoRA を適用。
    lora["target_modules"] が未指定/不正なら自動推定に置き換える。
    """
    if not lora:
        return False
    try:
        from peft import LoraConfig, get_peft_model
    except Exception:
        return False

    base = model_with_vhead.pretrained_model

    # ユーザ指定が実在しなければ自動推定に切り替え
    user_targets = lora.get("target_modules")
    if user_targets:
        names = [n for n, _ in base.named_modules()]
        if not any(any(t in n for n in names) for t in user_targets):
            target_modules = _guess_lora_targets(base)
        else:
            target_modules = user_targets
    else:
        target_modules = _guess_lora_targets(base)

    from peft import LoraConfig  # 再保証
    cfg = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias=str(lora.get("bias", "none")),
        task_type=str(lora.get("task_type", "CAUSAL_LM")),
        target_modules=target_modules,
    )

    from peft import get_peft_model
    peft_base = get_peft_model(base, cfg)
    model_with_vhead.pretrained_model = peft_base
    try:
        peft_base.print_trainable_parameters()
    except Exception:
        pass
    return True


# -----------------------------
# 生成&保存の薄いラッパ
# -----------------------------
class _GenWrapper:
    def __init__(self, trainer: PPOTrainer, tokenizer, gen_defaults: Optional[Dict[str, Any]]):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.gen_defaults = gen_defaults or {}
        self.accelerator = trainer.accelerator  # 互換のため公開

    @property
    def model(self):
        return self.trainer.model

    def step(self, *args, **kwargs):
        return self.trainer.step(*args, **kwargs)

    def save_pretrained(self, out_dir: str):
        try:
            self.trainer.model.save_pretrained(out_dir)
        except Exception:
            base = self.accelerator.unwrap_model(self.trainer.model)
            base.save_pretrained(out_dir)

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        """
        prompts: List[str] -> List[str]
        ValueHead ラッパの内側 (pretrained_model) で生成する
        """
        kw = dict(self.gen_defaults)
        kw.update(kwargs or {})

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.trainer.accelerator.device)

        gen_model = getattr(self.trainer.model, "pretrained_model", self.trainer.model)

        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                do_sample=True,
                top_p=kw.get("top_p"),
                top_k=kw.get("top_k"),
                temperature=kw.get("temperature"),
                repetition_penalty=kw.get("repetition_penalty"),
                min_new_tokens=kw.get("min_new_tokens"),
                max_new_tokens=kw.get("max_new_tokens"),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.batch_decode(out, skip_special_tokens=True)


# -----------------------------
# エクスポート関数
# -----------------------------
def build_ppo_trainer(
    model_name_or_path: str,
    *,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    mini_batch_size: int = 4,
    ppo_epochs: int = 4,
    kl_target: float = 0.1,
    max_grad_norm: float = 1.0,
    ratio_threshold: float = 0.2,
    trust_remote_code: bool = False,
    lora: Optional[Dict[str, Any]] = None,
    gen_defaults: Optional[Dict[str, Any]] = None,
) -> _GenWrapper:
    """
    TRL の PPOTrainer を初期化して _GenWrapper を返す。
    - ValueHead 付きでロード
    - LoRA（任意）を内側に適用（層は自動推定）
    - 参照モデルは create_reference_model で生成
    """
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ★ decoder-only 用：左パディングを推奨
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    # モデル（ValueHeadつき）
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name_or_path,
        dtype=dtype,  # torch_dtype は非推奨
        trust_remote_code=trust_remote_code,
    )

    # LoRA（任意）を内側に適用
    _maybe_apply_lora_to_inner(model, lora)

    # 参照モデル（KL用）
    ref_model = create_reference_model(model)

    # PPO 設定
    cfg = PPOConfig(
        model_name=model_name_or_path,
        learning_rate=learning_rate,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        target_kl=kl_target,
        max_grad_norm=max_grad_norm,
        ratio_threshold=ratio_threshold,
        remove_unused_columns=False,
        log_with="wandb",
    )

    # Trainer
    trainer = PPOTrainer(
        config=cfg,
        model=model,          # ValueHead ラッパ
        ref_model=ref_model,  # 参照もラッパ
        tokenizer=tokenizer,
        dataset=None,         # 手動ロールアウト
    )

    return _GenWrapper(trainer, tokenizer, gen_defaults or {})

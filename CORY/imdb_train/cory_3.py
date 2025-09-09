# coding=utf-8
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, List, Any

import torch
import tyro
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

# ---------------- GPU pick ----------------
def select_least_used_gpu():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode("utf-8")
    mem = [int(x) for x in out.strip().split("\n")]
    return mem.index(min(mem))

gpu = select_least_used_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
print(f">> Using GPU: {gpu}")

# ---------------- Args ----------------
@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="gpt2-medium",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            # 小さめで安定寄り
            learning_rate=5e-6,
            batch_size=8,
            mini_batch_size=8,
            gradient_accumulation_steps=1,
            early_stopping=False,
            # クリップを強めに
            target_kl=0.1,
            kl_penalty="full",
            init_kl_coef=0.05,
            seed=123,
            # logging
            log_with="wandb",
            tracker_project_name="imdb-ppo",
            tracker_kwargs={
                "wandb": {
                    "entity": os.environ.get("WANDB_ENTITY", None),
                    "name": f"imdb-dual-role-ppo-lora-{time.strftime('%m%d%H%M')}",
                }
            },
        )
    )
    use_peft: bool = True
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(r=16, lora_alpha=16, bias="none", task_type="CAUSAL_LM")
    )

args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

# ---------------- Dataset ----------------
def build_dataset(config, query_dataset, min_len=2, max_len=8):
    tok = AutoTokenizer.from_pretrained(config.model_name)
    tok.pad_token = tok.eos_token
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    sampler = LengthSampler(min_len, max_len)

    def tokenize(sample):
        sample["input_ids"] = tok.encode(sample["review"])[: sampler()]
        sample["query"] = tok.decode(sample["input_ids"])
        return sample

    return ds.map(tokenize, batched=False)

dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)

def collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    queries = [f["query"] for f in features]
    return {"input_ids": input_ids, "query": queries}

# ---------------- Model / Trainer ----------------
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.ppo_config.model_name, peft_config=args.peft_config
)
tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

ppo_trainer = PPOTrainer(
    args.ppo_config, model, None, tokenizer, dataset=dataset, data_collator=collator
)

# ---------------- Reward pipeline ----------------
task, reward_model_name = args.ppo_config.reward_model.split(":")
sentiment_pipe = pipeline(task, model=reward_model_name, device=ppo_trainer.accelerator.device)

# transformers>=4.41 は return_all_scores 推奨が top_k=None に変更
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 32}

def positive_score(example_out: Any) -> torch.Tensor:
    # list[{"label": "...", "score": ...}, ...] または dict{"label": "...", "score": ...}
    if isinstance(example_out, list):
        for d in example_out:
            lab = str(d.get("label", "")).upper()
            if "POS" in lab:
                return torch.tensor(float(d.get("score", 0.0)), dtype=torch.float32)
        for d in example_out:
            lab = str(d.get("label", "")).upper()
            if "NEG" in lab:
                return torch.tensor(1.0 - float(d.get("score", 0.0)), dtype=torch.float32)
        best = max(example_out, key=lambda x: float(x.get("score", 0.0)))
        return torch.tensor(float(best.get("score", 0.0)), dtype=torch.float32)
    if isinstance(example_out, dict):
        lab = str(example_out.get("label", "")).upper()
        s = float(example_out.get("score", 0.0))
        return torch.tensor(s if "POS" in lab else (1.0 - s), dtype=torch.float32)
    return torch.tensor(0.0, dtype=torch.float32)

def pos_score_from_pipeline(outputs: List[Any]) -> List[torch.Tensor]:
    return [positive_score(o) for o in outputs]

# ---------------- Generation settings ----------------
# 生成崩壊対策の一例（記号連発を抑制）
bad_excl_id = tokenizer.encode("!")[0]

BASE_GEN_KW = dict(
    # 長さ系は max_new_tokens のみ使用（min_length は渡さない）
    max_new_tokens=16,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    temperature=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    pad_token_id=tokenizer.pad_token_id,
    bad_words_ids=[[bad_excl_id]],
)

print(">> Training start (LoRA dual-role single-policy, stabilized).")

beta = 0.5          # 互いの影響度
reward_scale = 1.5  # スケーリング
output_length_sampler = LengthSampler(8, 16)  # 実際に使う新規トークン長

for step_idx, batch in tqdm(enumerate(ppo_trainer.dataloader), total=100, desc="Dual-Role PPO (LoRA)"):
    # ---------------- Observer 生成 ----------------
    q_obs_ids = batch["input_ids"]
    gen_kwargs = dict(BASE_GEN_KW)
    gen_kwargs["max_new_tokens"] = output_length_sampler()

    resp_obs_t = ppo_trainer.generate(q_obs_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    r_obs_list = tokenizer.batch_decode(resp_obs_t)

    # ---------------- Pioneer 生成（Observer をヒントに） ----------------
    merged_queries = [
        f"I can make this sentence '{q}{r}' more positive: {q}"
        for q, r in zip(batch["query"], r_obs_list)
    ]
    q_pio_ids = [torch.tensor(tokenizer.encode(m), dtype=torch.long) for m in merged_queries]

    resp_pio_t = ppo_trainer.generate(q_pio_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    r_pio_list = tokenizer.batch_decode(resp_pio_t)

    # ---------------- 報酬（まとめて推論して高速化） ----------------
    with torch.no_grad():
        both = sentiment_pipe(r_obs_list + r_pio_list, **sent_kwargs)
    out_obs, out_pio = both[: len(r_obs_list)], both[len(r_obs_list) :]

    s_obs = pos_score_from_pipeline(out_obs)
    s_pio = pos_score_from_pipeline(out_pio)

    # 役割ごとの報酬
    rew_obs = [torch.tanh((z_obs + beta * z_pio) / 2.0) * reward_scale for z_obs, z_pio in zip(s_obs, s_pio)]
    rew_pio = [torch.tanh((z_pio + beta * z_obs) / 2.0) * reward_scale for z_obs, z_pio in zip(s_obs, s_pio)]

    # ---------------- PPO 更新（役割ごとに別 step） ----------------
    stats_obs = ppo_trainer.step(q_obs_ids, resp_obs_t, rew_obs)
    stats_obs.pop("temp/average_new_tokens", None)
    ppo_trainer.log_stats(stats_obs, batch, rew_obs, columns_to_log=["query"])

    stats_pio = ppo_trainer.step(q_pio_ids, resp_pio_t, rew_pio)
    stats_pio.pop("temp/average_new_tokens", None)
    # Pioneer 側は query を置き換えた別バッチとして記録
    ppo_trainer.log_stats(
        stats_pio,
        {"query": merged_queries},
        rew_pio,
        columns_to_log=["query"],
    )

    if step_idx + 1 >= 100:
        break

print(">> Training finished.")

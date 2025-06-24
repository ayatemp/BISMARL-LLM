#!/usr/bin/env python3
# coding=utf-8
# -----------------------------------------------------------------------------
#  PPO fine-tuning example (single model) – “Script A” 完全版
#  * GPT-2-medium を IMDB レビューで RLHF (PPO) 微調整
#  * 左パディング／小さめバッチで VRAM を節約
#  * rescore: DistilBERT-IMDB (positive class score)
#  * 学習後に checkpoints/gpt2_ppo_final へ保存
# -----------------------------------------------------------------------------
import os, sys, subprocess, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch, tyro, wandb
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, pipeline, set_seed
from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
)
from trl.core import LengthSampler
from peft import LoraConfig

# -----------------------------------------------------------------------------#
# 0.  GPU を自動選択（メモリ使用量が最少のもの）                               #
# -----------------------------------------------------------------------------#
def _pick_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode()
        mem = [int(x) for x in out.strip().split("\n")]
        return str(mem.index(min(mem)))
    except Exception:
        return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = _pick_gpu()
print(f"[INFO] CUDA_VISIBLE_DEVICES set to {os.environ['CUDA_VISIBLE_DEVICES']}")

# -----------------------------------------------------------------------------#
# 1.  解析用引数                                                               #
# -----------------------------------------------------------------------------#
@dataclass
class ScriptArguments:
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            tracker_kwargs={
                "wandb": {
                    "entity": "ayato-kaku-",
                    "name"  : f"single-rl-gpt2-medium-{time.strftime('%m%d%H%M')}",
                }
            },
            tracker_project_name="imdb_ppo",
            model_name   ="D:/LLMModel/gpt2-medium",
            query_dataset="D:/LLMModel/imdb_dataset",
            reward_model ="sentiment-analysis:D:/LLMModel/distilbert-imdb",
            learning_rate=1.41e-5,
            log_with="wandb",
            batch_size=64,
            mini_batch_size=64,
            gradient_accumulation_steps=1,
            early_stopping=False,
            kl_penalty="full",
            seed=123,
            init_kl_coef=0.3,
        )
    )
    use_seq2seq: bool = False
    use_peft   : bool = False
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(r=16, lora_alpha=16, bias="none", task_type="CAUSAL_LM")
    )
    trust_remote_code: bool = False
    group: Optional[str] = "imdb-single"

args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

# -----------------------------------------------------------------------------#
# 2.  Tokenizer – decoder-only なので left padding                              #
# -----------------------------------------------------------------------------#
tokenizer = AutoTokenizer.from_pretrained(
    args.ppo_config.model_name, trust_remote_code=args.trust_remote_code
)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
print(f"[INFO] tokenizer padding_side = {tokenizer.padding_side}")

# -----------------------------------------------------------------------------#
# 3.  データセット構築                                                         #
# -----------------------------------------------------------------------------#
def build_dataset(cfg: PPOConfig, path: str, min_len=2, max_len=8):
    if os.path.isdir(path):
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
    else:
        ds = load_dataset(path, split="train")

    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 50)          # ─ 長さフィルタ

    sampler = LengthSampler(min_len, max_len)

    def _tok(ex):
        ids = tokenizer.encode(ex["review"])[: sampler()]
        ex["input_ids"] = ids
        ex["query"]     = tokenizer.decode(ids)
        return ex

    ds = ds.map(_tok, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds

print("[INFO] Loading dataset …")
dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)
print(f"[INFO] dataset size = {len(dataset)}")
if len(dataset) == 0:
    sys.exit("[ERR] Dataset is empty – adjust filter condition.")

# -----------------------------------------------------------------------------#
# 4.  Collator (left pad & attention_mask)                                      #
# -----------------------------------------------------------------------------#
def _left_pad(ids: torch.Tensor, pad_id: int, tgt_len: int) -> torch.Tensor:
    diff = tgt_len - ids.size(0)
    return ids if diff <= 0 else torch.nn.functional.pad(ids, (diff, 0), value=pad_id)

def make_collator(tok):
    pad_id = tok.pad_token_id
    def _collate(batch):
        seqs   = [b["input_ids"] for b in batch]
        maxlen = max(len(s) for s in seqs)
        ids    = torch.stack([_left_pad(s, pad_id, maxlen) for s in seqs])
        mask   = (ids != pad_id).long()
        return {"input_ids": ids, "attention_mask": mask, "query": [b["query"] for b in batch]}
    return _collate

collator = make_collator(tokenizer)

# -----------------------------------------------------------------------------#
# 5.  モデル & PPOTrainer                                                      #
# -----------------------------------------------------------------------------#
ModelCls = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
model = ModelCls.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)

ppo_trainer = PPOTrainer(
    args.ppo_config, model, None, tokenizer,
    dataset=dataset, data_collator=collator
)
ppo_trainer.tokenizer.padding_side = "left"

# -----------------------------------------------------------------------------#
# 6.  Reward (sentiment) pipeline                                              #
# -----------------------------------------------------------------------------#
task, reward_name = args.ppo_config.reward_model.split(":", 1)
sentiment_pipe = pipeline(
    task, model=reward_name,
    device=ppo_trainer.accelerator.device,
    function_to_apply="none",
    batch_size=16
)
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
sentiment_pipe.tokenizer.padding_side = "left"

# -----------------------------------------------------------------------------#
# 7.  PPO 学習ループ                                                           #
# -----------------------------------------------------------------------------#
sampler = LengthSampler(4, 16)
generation_kwargs = dict(
    min_length = -1,
    top_k      = 0.0,
    top_p      = 1.0,
    do_sample  = True,
    pad_token_id = tokenizer.eos_token_id,
    repetition_penalty = 1.1,
)

device_t = ppo_trainer.accelerator.device

for step, batch in tqdm(
    enumerate(ppo_trainer.dataloader, 1),
    total=len(ppo_trainer.dataloader),
    dynamic_ncols=True
):
    query_tensors = [ids for ids in batch["input_ids"]]

    generation_kwargs["max_new_tokens"] = sampler()
    resp, resp_ref = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )

    batch["response"]      = tokenizer.batch_decode(resp)
    batch["ref_response"]  = tokenizer.batch_decode(resp_ref)

    # ---- reward -------------------------------------------------------------
    texts   = [q + r for q, r in zip(batch["query"], batch["response"])]
    rewards = [torch.tensor(out["score"], device=device_t)
               for out in sentiment_pipe(texts)]

    ref_txt = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_rwd = [torch.tensor(out["score"], device=device_t)
               for out in sentiment_pipe(ref_txt)]
    batch["ref_rewards"] = ref_rwd

    # ---- PPO step ----------------------------------------------------------
    stats = ppo_trainer.step(query_tensors, resp, rewards)
    ppo_trainer.log_stats(stats, batch, rewards,
                          columns_to_log=["query", "response"])

# -----------------------------------------------------------------------------#
# 8.  学習済みモデル保存                                                       #
# -----------------------------------------------------------------------------#
save_dir = Path("checkpoints/gpt2_ppo_final")
save_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"[INFO] model saved to {save_dir.resolve()}")

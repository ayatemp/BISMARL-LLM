# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------
# Duo-PPO (Extensive-Game) for IMDB sentiment – self-contained version
# ----------------------------------------------------------------------------------------
import os, sys, subprocess, time, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import torch, tyro, wandb
from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import pipeline, AutoTokenizer, set_seed
from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)
from trl.core import LengthSampler
from peft import LoraConfig

# ----------------------- ここだけ変数で簡単に切り替えられる ------------------------------
DEBUG_MAX_STEPS = None       # 例: 100 でデバッグ停止 / None で最後まで
BATCH_SIZE      = 64         # GPU VRAM に合わせて
MIN_QUERY_LEN   = 2          # 入力文を切り詰める長さ範囲（token 単位）
MAX_QUERY_LEN   = 8
REWARD_BATCH    = 32         # sentiment_pipe のバッチサイズ
# ----------------------------------------------------------------------------------------

# -------------------------------------------------------------------------
# 0. GPU を自動で決定（なければ CPU）
# -------------------------------------------------------------------------
def pick_gpu() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode()
        mem = [int(x) for x in out.strip().splitlines()]
        return str(mem.index(min(mem)))
    except Exception:
        return ""

gpu_id = pick_gpu()
if gpu_id:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"[INFO] CUDA_VISIBLE_DEVICES = {gpu_id}")
else:
    print("[WARN] No NVIDIA GPU detected – running on CPU")

# -------------------------------------------------------------------------
# 1. コマンドライン引数
# -------------------------------------------------------------------------
@dataclass
class Args:
    # ★── 自前モデル / データセットパスを設定
    model_name: str = "D:/LLMModel/gpt2-medium"                     # ★
    dataset_path: str = "D:/LLMModel/imdb_dataset"                  # ★ (load_dataset で使える名前かローカル dir)
    reward_model: str = "sentiment-analysis:D:/LLMModel/distilbert-imdb"  # ★ task:model
    project: str = "imdb_CORY"
    entity: str  = "ayato-kaku-"
    seed: int = 42
    use_seq2seq: bool = False
    use_peft: bool = False
    trust_remote_code: bool = False

    # swap & eval
    swap_freq: int = 5
    eval_freq: int = 7

    # PEFT 設定（使う場合だけ）
    peft_r: int = 16
    peft_alpha: int = 16

args = tyro.cli(Args)
set_seed(args.seed)

# -------------------------------------------------------------------------
# 2. Tokenizer
# -------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
print(f"[INFO] tokenizer padding_side = {tokenizer.padding_side}")

# -------------------------------------------------------------------------
# 3. データセット
# -------------------------------------------------------------------------
def build_dataset(path: str) -> "datasets.Dataset":
    if os.path.isdir(path):
        ds = load_from_disk(path)
        if isinstance(ds, DatasetDict):
            ds = ds["train"]
    else:
        ds = load_dataset(path, split="train")
    ds = ds.rename_columns({ds.column_names[0]: "review"})  # text → review 等
    ds = ds.filter(lambda x: len(x["review"]) > 50)

    sampler = LengthSampler(MIN_QUERY_LEN, MAX_QUERY_LEN)

    def _tok(ex):
        ids = tokenizer.encode(ex["review"])[: sampler()]
        ex["input_ids"] = ids
        ex["query"] = tokenizer.decode(ids)
        return ex

    # Windows → num_proc=1 で spawn 問題回避
    ds = ds.map(_tok, batched=False)
    ds.set_format(type="torch", columns=["input_ids", "query"])
    return ds

print("[INFO] Loading dataset …")
dataset = build_dataset(args.dataset_path)
print(f"[INFO] dataset size = {len(dataset)}")
if not len(dataset):
    sys.exit("Empty dataset – check path / filter condition.")

def collate_fn(batch):
    pad = tokenizer.pad_token_id
    seqs = [b["input_ids"] for b in batch]
    max_len = max(len(s) for s in seqs)
    ids  = torch.stack([torch.nn.functional.pad(s, (max_len-len(s), 0), value=pad) for s in seqs])
    mask = (ids != pad).long()
    return {"input_ids": ids, "attention_mask": mask, "query": [b["query"] for b in batch]}

# -------------------------------------------------------------------------
# 4. モデル準備
# -------------------------------------------------------------------------
ModelCls = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead
peft_cfg = None
if args.use_peft:
    peft_cfg = LoraConfig(r=args.peft_r, lora_alpha=args.peft_alpha, bias="none", task_type="CAUSAL_LM")

def build_trainer(run_name: str) -> PPOTrainer:
    cfg = PPOConfig(
        tracker_project_name=args.project,
        tracker_kwargs={"wandb": {"entity": args.entity, "name": run_name}},
        model_name=args.model_name,
        log_with="wandb",
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        seed=args.seed,
        kl_penalty="full",
        init_kl_coef=0.3,
    )
    model = ModelCls.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        peft_config=peft_cfg,
    )
    trainer = PPOTrainer(cfg, model, None, tokenizer, dataset=dataset, data_collator=collate_fn)
    trainer.tokenizer.padding_side = "left"
    return trainer

trainer1 = build_trainer("LLM1")
trainer2 = build_trainer("LLM2")

device = trainer1.accelerator.device

# -------------------------------------------------------------------------
# 5. 報酬モデル
# -------------------------------------------------------------------------
task, reward_model_path = args.reward_model.split(":", 1)
sentiment_pipe = pipeline(task, model=reward_model_path,
                          device=device if torch.cuda.is_available() else -1,
                          batch_size=REWARD_BATCH,
                          function_to_apply="none")
sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
sentiment_pipe.tokenizer.padding_side = "left"

# -------------------------------------------------------------------------
# 6. PPO ループ
# -------------------------------------------------------------------------
merge_tpl = 'I can make this sentence "{}" more positive: {}'
gen_len_sampler = LengthSampler(4, 16)
gen_kwargs = dict(
    min_length=-1, top_k=0, top_p=1, do_sample=True,
    pad_token_id=tokenizer.eos_token_id, repetition_penalty=1.1,
)

swap = False
loop = tqdm(enumerate(trainer1.dataloader, 1), total=len(trainer1.dataloader))
for step, batch in loop:
    # ---------- LLM-1 ----------
    gen_kwargs["max_new_tokens"] = gen_len_sampler()
    resp1, _ = trainer1.generate(batch["input_ids"], return_prompt=False, generate_ref_response=False, **gen_kwargs)
    txt_resp1 = tokenizer.batch_decode(resp1)

    # ---------- LLM-2 ----------
    merged_queries = [merge_tpl.format(q + r, q) for q, r in zip(batch["query"], txt_resp1)]
    merged_ids    = [torch.tensor(tokenizer.encode(mq)) for mq in merged_queries]
    resp2, _ = trainer2.generate(merged_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    txt_resp2 = tokenizer.batch_decode(resp2)

    # ---------- 報酬 ----------
    rew1 = [torch.tensor(o["score"], device=device) for o in sentiment_pipe([q + r for q, r in zip(batch["query"], txt_resp1)])]
    rew2 = [torch.tensor(o["score"], device=device) for o in sentiment_pipe([q + r for q, r in zip(batch["query"], txt_resp2)])]

    # cooperative game (合計報酬)
    coop1 = [r1 + r2 for r1, r2 in zip(rew1, rew2)]
    coop2 = [r1 + r2 for r1, r2 in zip(rew1, rew2)]

    stats1 = trainer1.step(batch["input_ids"], resp1, coop1)
    stats2 = trainer2.step(merged_ids,       resp2, coop2)

    # ---------- ログ ----------
    log_batch = {
        "query": batch["query"],
        "resp1": txt_resp1,
        "resp2": txt_resp2,
    }
    trainer1.log_stats(stats1, log_batch, rew1)

    # ---------- Swap ----------
    if step % args.swap_freq == 0:
        swap = not swap
        trainer1, trainer2 = trainer2, trainer1  # オブジェクトを丸ごと交換
        loop.set_description(f"[SWAP at step {step}]")

    # ---------- デバッグ早期終了 ----------
    if DEBUG_MAX_STEPS and step >= DEBUG_MAX_STEPS:
        print(f"[INFO] Stopping after {DEBUG_MAX_STEPS} steps (debug mode)")
        break

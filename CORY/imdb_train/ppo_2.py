# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0

import sys
sys.path.append('/home/trl/trl')

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import tyro
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    PPOConfig,
    PPOTrainer,
    set_seed,
)
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available

tqdm.pandas()


def select_least_used_gpu():
    smi_output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']
    ).decode('utf-8')
    gpu_memory = [int(x) for x in smi_output.strip().split('\n')]
    least_used_gpu = gpu_memory.index(min(gpu_memory))
    return least_used_gpu


# 必要なら手動で指定する（.pyの外=ターミナルでCUDA_VISIBLE_DEVICESを設定推奨）
print(">> Chosen GPU:", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))

@dataclass
class ScriptArguments:
    # --- ここのPPOConfigは安定寄りに調整（必要に応じて戻してください） ---
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            tracker_project_name=os.environ.get("WANDB_PROJECT", None),
            log_with="wandb",
            tracker_kwargs={
                "wandb": {
                    "entity": os.environ.get("WANDB_ENTITY", None),
                    "name": f"imdb-ppo-{time.strftime('%m%d%H%M', time.localtime())}",
                }
            },

            model_name="gpt2-medium",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",

            # ★安定化寄り設定（必要なら戻す）
            learning_rate=1e-5,              # 5e-5 → 1e-5
            mini_batch_size=8,
            batch_size=32,                   # 64 → 32
            gradient_accumulation_steps=1,
            early_stopping=False,
            kl_penalty="kl",                 # "full" → "kl"
            init_kl_coef=1.0,                # 0.3 → 1.0
            # target_kl は TRL のバージョンによっては無視される場合あり
            target_kl=0.1,

            seed=123,
            use_score_scaling=True,          # 有効化
            use_score_norm=True,             # 有効化
            score_clip=None,
        )
    )
    use_seq2seq: bool = False
    use_peft: bool = False
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    group: Optional[str] = field(default="imdb-single", metadata={"help": "Wandb grouping"})


args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

# sentiment pipeline の設定（deprecation回避）
# return_all_scores は非推奨なので top_k=None を使う
sent_kwargs = {"top_k": None, "function_to_apply": "softmax", "batch_size": 16}

trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


def build_dataset(config, query_dataset, input_min_text_length=2, input_max_text_length=8):
    """
    IMDBから短いプロンプト（query）を作る。
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # 長文レビューの頭から少数トークンだけをqueryとして切り出す
        ids: List[int] = tokenizer.encode(sample["review"])
        ids = ids[: input_size()]
        sample["input_ids"] = ids
        sample["query"] = tokenizer.decode(ids)
        return sample

    ds = ds.map(tokenize, batched=False)
    return ds


dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)


def collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    queries = [f["query"] for f in features]
    return {"input_ids": input_ids, "query": queries}


# モデル・参照モデル・トークナイザー
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(
        args.ppo_config.model_name,
        trust_remote_code=args.trust_remote_code
    )
    device_map = None
    peft_config = None
else:
    peft_config = args.peft_config
    ref_model = None
    device_map = {"": Accelerator().local_process_index}

model = trl_model_class.from_pretrained(
    args.ppo_config.model_name,
    trust_remote_code=args.trust_remote_code,
    device_map=device_map,
    peft_config=peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

ppo_trainer = PPOTrainer(
    args.ppo_config,
    model,
    ref_model,
    tokenizer,
    dataset=dataset,
    data_collator=collator
)

# sentiment pipeline（PPOのdeviceに合わせる）
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        pipe_device = "xpu:0"
    elif is_npu_available():
        pipe_device = "npu:0"
    else:
        pipe_device = 0 if torch.cuda.is_available() else "cpu"
else:
    # 分散時はとりあえず rank 0 GPU に置くのが無難（必要に応じ調整）
    pipe_device = 0 if torch.cuda.is_available() else "cpu"

ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, model_name = args.ppo_config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=model_name, device=pipe_device)
else:
    sentiment_pipe = pipeline(task, model=model_name, device=pipe_device)

# pad_token の保護
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# 生成長
output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": 8,          # 初期は短め固定でもOK
    "top_k": 50,
    "top_p": 0.9,
    "temperature": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "repetition_penalty": 1.1,
}

# POSITIVE のインデックスを取得（ラベル順がモデルにより異なるため動的に）
id2label = sentiment_pipe.model.config.id2label  # 例: {0:'NEGATIVE', 1:'POSITIVE'}
pos_candidates = [i for i, lab in id2label.items() if str(lab).upper().startswith("POS")]
if not pos_candidates:
    # 念のためフォールバック（2クラス想定で1番目をPOS扱い）
    pos_index = 1
else:
    pos_index = pos_candidates[0]

# --- 学習ループ ---
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

    query_tensors = batch["input_ids"]

    # 可変長にしたい場合はサンプラー使用（初期は固定でも良い）
    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len

    # 生成（参照応答も取得）
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=False,
        generate_ref_response=True,
        **generation_kwargs
    )

    # テキスト化（特殊トークン除去）
    batch["response"] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors, skip_special_tokens=True)

    texts = batch["response"]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)  # top_k=None, softmax

    # list[Tensor]（各(1,)）を “PPOのdevice” に合わせて用意
    pos_probs = [torch.tensor(o[pos_index]["score"], dtype=torch.float32, device=device) for o in pipe_outputs]

    # ====== ここから：ログ用の「生の報酬」統計（z-normalize 前） ======
    raw_rewards_tensor = torch.stack(pos_probs)  # shape: (B,)
    # 分散のバイアス補正なし（W&B 表示を安定させたいので unbiased=False）
    raw_mean = raw_rewards_tensor.mean().item()
    raw_std  = raw_rewards_tensor.std(unbiased=False).item()
    raw_min  = raw_rewards_tensor.min().item()
    raw_max  = raw_rewards_tensor.max().item()

    # 単GPU前提ならそのまま wandb.log でOK（分散なら rank==0 のときだけにする）
    wandb.log({
        "env/raw_reward_mean": raw_mean,
        "env/raw_reward_std":  raw_std,
        "env/raw_reward_min":  raw_min,
        "env/raw_reward_max":  raw_max,
    })
    # ====== ここまで：生の報酬ログ ======

    # PPO に渡すのは z-score 正規化後
    rewards_tensor = (raw_rewards_tensor - raw_rewards_tensor.mean()) / (raw_rewards_tensor.std(unbiased=False) + 1e-8)

    # list[Tensor] 各要素 shape=(1,)
    rewards = [r.unsqueeze(0).detach() for r in rewards_tensor]

    # 参考までに：正規化後の統計もログ（reward_std が ~1 になるはず）
    wandb.log({
        "env/norm_reward_mean": rewards_tensor.mean().item(),
        "env/norm_reward_std":  rewards_tensor.std(unbiased=False).item(),
        "env/norm_reward_min":  rewards_tensor.min().item(),
        "env/norm_reward_max":  rewards_tensor.max().item(),
    })

    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    try:
        stats.pop("temp/average_new_tokens")
    except KeyError:
        pass

    # ログ（存在するキーだけ）
    ppo_trainer.log_stats(
        stats,
        batch,
        rewards,
        columns_to_log=["query", "response"],
    )

    # デバッグしたければ早期break外す
    if epoch > 100:
        break

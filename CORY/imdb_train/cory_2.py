# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.

import os
import sys
import time
import random
import subprocess
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
    PPOConfig,
    set_seed,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)
from trl.core import LengthSampler

# ====== ユーティリティ（あなたの utils を使わない最小構成） ============================
# もともとの Model_Generator 等が手元にある場合はそちらを使ってOKです。
# ここでは self-contained に PPOTrainer を作る関数だけ用意します。
from trl import PPOTrainer

def make_ppo_trainer(
    ppo_cfg: PPOConfig,
    model_name_or_path: str,
    tokenizer: AutoTokenizer,
    dataset,
    data_collator,
    trust_remote_code: bool = False,
    peft_config: Optional[LoraConfig] = None,
    ref_model: Optional[torch.nn.Module] = None,
    device_map=None,
    use_seq2seq: bool = False,
):
    model_cls = AutoModelForSeq2SeqLMWithValueHead if use_seq2seq else AutoModelForCausalLMWithValueHead
    model = model_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
    )
    trainer = PPOTrainer(
        ppo_cfg,
        model,
        ref_model,
        tokenizer,
        dataset=dataset,
        data_collator=data_collator,
    )
    device = trainer.accelerator.device
    return model, trainer, device

# ====== GPU 選択（安全） =========================================================
def select_least_used_gpu() -> Optional[int]:
    try:
        smi_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode("utf-8")
        gpu_memory = [int(x) for x in smi_output.strip().split("\n")]
        return int(gpu_memory.index(min(gpu_memory)))
    except Exception:
        return None  # GPUが無ければCPU

gpu_idx = select_least_used_gpu()
if gpu_idx is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
print(f">> Chosen GPU: {os.environ.get('CUDA_VISIBLE_DEVICES','cpu')}")

# ====== 引数 ======================================================================
random_name = str(random.random()).split(".")[1]
exp_class = "extensive-game-" + random_name

# base_name = "/home/trl/trl/hf_hub/models/gpt2-"
base_name = "imdb_train/temp_models/gpt2-"
temp_model_name_1 = base_name + exp_class + "-1"
temp_model_name_2 = base_name + exp_class + "-2"
os.makedirs(temp_model_name_1, exist_ok=True)
os.makedirs(temp_model_name_2, exist_ok=True)

@dataclass
class ScriptArguments:
    # ベースモデル
    model_name: str = "gpt2-medium"

    # PPOConfig（安定寄り）
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            tracker_project_name=os.environ.get("WANDB_PROJECT", "imdb-ppo"),
            log_with="wandb",
            tracker_kwargs={
                "wandb": {
                    "entity": os.environ.get("WANDB_ENTITY", None),
                    "name": f"imdb-duo-ppo-{time.strftime('%m%d%H%M', time.localtime())}",
                    "group": "imdb-extensive-game",
                }
            },
            model_name="gpt2-medium",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
            learning_rate=1e-5,                  # 5e-5 → 1e-5（ratio暴騰抑制）
            mini_batch_size=8,                   # 安定寄り
            batch_size=32,                       # 64→32（KL制御を効かせやすく）
            gradient_accumulation_steps=1,
            early_stopping=False,
            kl_penalty="kl",                     # "full"→"kl"（TRLの新推奨。安定化）
            target_kl=0.1,                       # 追加：KL目標
            init_kl_coef=0.2,                    # 0.3→0.2（強すぎ回避）
            seed=123,
            use_score_scaling=False,
            use_score_norm=True,                 # バッチ内正規化ON（安定）
            score_clip=None,
        )
    )

    # モデル種別など
    use_seq2seq: bool = False
    use_peft: bool = False
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(r=16, lora_alpha=16, bias="none", task_type="CAUSAL_LM")
    )
    trust_remote_code: bool = False

    # duo まわり
    eval_freq: int = 7
    swap_freq: int = 5
    reward_type: str = "independent"  # ここでは cooperative 的に合成

    # 一時保存先
    temp_model_name_1: str = temp_model_name_1
    temp_model_name_2: str = temp_model_name_2

args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

# ====== データセット構築（右パディング前提＆短すぎる行除去） ==============================
def build_dataset(config, query_dataset, input_min_text_length=8, input_max_text_length=64):
    tok = AutoTokenizer.from_pretrained(config.model_name)
    tok.pad_token = tok.eos_token

    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200)  # 短文は除外

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # 先頭から input_size() トークン相当を query として使う
        ids = tok.encode(sample["review"])
        sl = input_size()
        sample["input_ids"] = ids[:sl]
        sample["query"] = tok.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize)
    return ds

dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)

def collator(features):
    # TRL は list[tensor] を受け取れる。右パディングは内部で処理される。
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    queries = [f["query"] for f in features]
    return {"input_ids": input_ids, "query": queries}

# ====== トークナイザ & PPOトレーナ2体 =================================================
tokenizer = AutoTokenizer.from_pretrained(args.ppo_config.model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

ref_model = None
device_map = None
peft_cfg = args.peft_config if args.use_peft else None
trl_model_class = AutoModelForSeq2SeqLMWithValueHead if args.use_seq2seq else AutoModelForCausalLMWithValueHead

model_1, ppo_trainer_1, device_1 = make_ppo_trainer(
    args.ppo_config, args.ppo_config.model_name, tokenizer, dataset, collator,
    trust_remote_code=args.trust_remote_code, peft_config=peft_cfg, ref_model=ref_model, device_map=device_map,
    use_seq2seq=args.use_seq2seq,
)
model_2, ppo_trainer_2, device_2 = make_ppo_trainer(
    args.ppo_config, args.ppo_config.model_name, tokenizer, dataset, collator,
    trust_remote_code=args.trust_remote_code, peft_config=peft_cfg, ref_model=ref_model, device_map=device_map,
    use_seq2seq=args.use_seq2seq,
)

# ====== 報酬モデル（pipeline） ====================================================
# return_all_scores は deprecated。top_k=None で全ラベル取得→POSITIVE抽出。
task, rm_name = args.ppo_config.reward_model.split(":")
sentiment_pipe = pipeline(task, model=rm_name, device=0 if torch.cuda.is_available() else -1, top_k=None)

# pad_token を保証
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if getattr(sentiment_pipe.model.config, "pad_token_id", None) is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# ====== 生成（安定寄り） ==========================================================
output_len_sampler = LengthSampler(16, 48)  # 短すぎるとRMが安定しないため少し伸ばす
gen_kwargs = {
    "min_length": -1,
    "do_sample": True,
    "top_k": 0,               # nucleus only
    "top_p": 0.95,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "repetition_penalty": 1.05,
}

merge_template = 'Rewrite this review continuation more positive while staying coherent:\nInput: "{}"\nOutput: '

# ====== 報酬ユーティリティ =========================================================
def positive_prob_from_pipeline_outputs(outputs: List[List[dict]]) -> torch.Tensor:
    # outputs: list (batch) of list-of-dicts [{"label":"NEGATIVE","score":...},{"label":"POSITIVE","score":...}]
    probs = []
    for o in outputs:
        pos = 0.0
        for e in o:
            if str(e["label"]).upper().endswith("POSITIVE"):
                pos = float(e["score"])
                break
        probs.append(pos)
    return torch.tensor(probs, dtype=torch.float32)

def zscore(t: torch.Tensor) -> torch.Tensor:
    mu = t.mean()
    sigma = t.std(unbiased=False)
    if sigma < 1e-8:
        return torch.zeros_like(t)
    return (t - mu) / sigma

# ====== ループ =====================================================================
swap = False
dataloader = ppo_trainer_1.dataloader

# 参考：コードバックアップをW&Bに上げたいとき
# arti_code = wandb.Artifact("code", type="code")
# arti_code.add_file(__file__)
# wandb.log_artifact(arti_code)

for step, batch in tqdm(enumerate(dataloader), desc="Duo-PPO"):
    # --- Observer (LLM1)
    query_tensors = batch["input_ids"]
    gen_len = output_len_sampler()
    gen_kwargs["max_new_tokens"] = gen_len

    resp_tensors_1, ref_resp_tensors = ppo_trainer_1.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **gen_kwargs
    )
    batch["response_llm1"] = tokenizer.batch_decode(resp_tensors_1)
    batch["ref_response"] = tokenizer.batch_decode(ref_resp_tensors)

    # --- Pioneer (LLM2): LLM1出力を使って“よりポジに”生成
    merged_queries = [merge_template.format(q + r) for q, r in zip(batch["query"], batch["response_llm1"])]
    merged_query_tensors = [torch.tensor(tokenizer.encode(mq), dtype=torch.long) for mq in merged_queries]

    resp_tensors_2 = ppo_trainer_2.generate(
        merged_query_tensors, return_prompt=False, generate_ref_response=False, **gen_kwargs
    )
    batch["response_llm2"] = tokenizer.batch_decode(resp_tensors_2)

    # --- 報酬（RM → POS prob → zscore → scale）
    rm_in_1 = [q + r for q, r in zip(batch["query"], batch["response_llm1"])]
    rm_out_1 = sentiment_pipe(rm_in_1)
    pos_1 = positive_prob_from_pipeline_outputs(rm_out_1)

    rm_in_2 = [q + r for q, r in zip(batch["query"], batch["response_llm2"])]
    rm_out_2 = sentiment_pipe(rm_in_2)
    pos_2 = positive_prob_from_pipeline_outputs(rm_out_2)

    # バッチ内 zscore で安定化しつつスケールは控えめ
    scale = 3.0
    rew_1 = zscore(pos_1) * scale
    rew_2 = zscore(pos_2) * scale

    # cooperative: 両者の合計を使う（独立のままでもOK）
    game_reward_1 = (rew_1 + rew_2).cpu().unbind()  # list[Tensor]
    game_reward_2 = (rew_1 + rew_2).cpu().unbind()

    # --- PPO step（ratio暴騰に備えてKLが効く設定）
    stats_1 = ppo_trainer_1.step(query_tensors, resp_tensors_1, list(game_reward_1))
    stats_2 = ppo_trainer_2.step(merged_query_tensors, resp_tensors_2, list(game_reward_2))

    # --- ログ整形（余計なキーは消す）
    for st in (stats_1, stats_2):
        if "temp/average_new_tokens" in st:
            st.pop("temp/average_new_tokens", None)

    # 2つの報酬内訳（可視化用）
    ppo_trainer_1.log_stats(
        stats_1,
        batch,
        list(game_reward_1),
        columns_to_log=["query", "response_llm1", "response_llm2", "ref_response"],
        LLM1_pos_prob=pos_1.tolist(),
        LLM2_pos_prob=pos_2.tolist(),
        reward_scale=scale,
    )

    # --- 役割入替（数 step ごと）
    if (step + 1) % args.swap_freq == 0:
        swap = not swap
        # 重いけど安全な「重みの入れ替え」
        model_1.save_pretrained(args.temp_model_name_1, push_to_hub=False)
        model_2.save_pretrained(args.temp_model_name_2, push_to_hub=False)

        # LLM1 ← temp2, LLM2 ← temp1
        model_1, ppo_trainer_1, device_1 = make_ppo_trainer(
            args.ppo_config, args.temp_model_name_2, tokenizer, dataset, collator,
            trust_remote_code=args.trust_remote_code, peft_config=peft_cfg, ref_model=ref_model, device_map=device_map,
            use_seq2seq=args.use_seq2seq,
        )
        model_2, ppo_trainer_2, device_2 = make_ppo_trainer(
            args.ppo_config, args.temp_model_name_1, tokenizer, dataset, collator,
            trust_remote_code=args.trust_remote_code, peft_config=peft_cfg, ref_model=ref_model, device_map=device_map,
            use_seq2seq=args.use_seq2seq,
        )

    # --- まずは 100 step で停止（検証用）
    if step + 1 >= 100:
        break

print("Training finished (100 steps).")

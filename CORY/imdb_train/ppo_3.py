# coding: utf-8
"""
IMDB PPO 学習 - 安定版 v2
- 右パディング / pad_token 明示
- dataset トークナイズ truncation=True
- nucleus-only 生成 (top_p=0.9, top_k=0)
- return_prompt=True ＋ 明示的 response_masks（生成トークンのみ更新）
- 報酬: 既定は POS 確率 * scale（符号を必ず正に）/ オプションで (POS-NEG)*scale
- KL をやや緩め (init_kl_coef/target_kl)
- 学習ステップ上限 = 100
"""

import os, time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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


print(">> Chosen GPU:", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))

# =========================
# 引数
# =========================
@dataclass
class ScriptArguments:
    # --- TRL PPO ---
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            tracker_project_name=os.environ.get("WANDB_PROJECT", "imdb-ppo"),
            log_with="wandb",
            tracker_kwargs={"wandb": {
                "entity": os.environ.get("WANDB_ENTITY", None),
                "name": f"imdb-ppo-{time.strftime('%m%d%H%M', time.localtime())}",
            }},
            model_name="gpt2-medium",
            query_dataset="imdb",
            reward_model="sentiment-analysis:lvwerra/distilbert-imdb",

            learning_rate=2e-5,
            mini_batch_size=8,
            batch_size=32,
            gradient_accumulation_steps=1,
            early_stopping=False,

            # KL をやや緩め
            kl_penalty="kl",
            init_kl_coef=0.1,
            target_kl=0.2,

            seed=123,
            use_score_scaling=True,
            use_score_norm=True,
            score_clip=None,
        )
    )
    # --- モデルモード ---
    use_seq2seq: bool = False
    use_peft: bool = False
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(r=16, lora_alpha=16, bias="none", task_type="CAUSAL_LM"),
    )
    trust_remote_code: bool = False
    group: Optional[str] = "imdb-single"

    # --- 実験オプション ---
    max_steps: int = 100                       # ここで 100 に制限
    reward_scale: float = 5.0                  # 報酬スケール（3〜10 で調整）
    reward_mode: str = "pos"                   # "pos" or "diff"  (diff は pos-neg)
    min_new_tokens: int = 8
    max_new_tokens: int = 32
    top_p: float = 0.9
    temperature: float = 1.0

args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

SENT_KW = {"top_k": None, "function_to_apply": "softmax", "batch_size": 16}
trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


# =========================
# データ作成
# =========================
def build_dataset(config: PPOConfig, query_dataset: str,
                  input_min_text_length: int = 2, input_max_text_length: int = 8):
    tok = AutoTokenizer.from_pretrained(config.model_name)
    tok.pad_token = tok.eos_token

    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    sampler = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        enc = tok(
            batch["review"],
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        )
        input_ids_list, query_list = [], []
        for ids in enc["input_ids"]:
            L = sampler()
            ids = ids[:L]
            input_ids_list.append(ids)
            query_list.append(tok.decode(ids, skip_special_tokens=True))
        return {"input_ids": input_ids_list, "query": query_list}

    ds = ds.map(tokenize, batched=True, batch_size=512)
    ds = ds.shuffle(seed=config.seed)
    return ds

dataset = build_dataset(args.ppo_config, args.ppo_config.query_dataset)

def collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    queries = [f["query"] for f in features]
    return {"input_ids": input_ids, "query": queries}


# =========================
# モデル
# =========================
if not args.use_peft:
    ref_model = trl_model_class.from_pretrained(args.ppo_config.model_name, trust_remote_code=args.trust_remote_code)
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
tokenizer.padding_side = "right"   # ★右パディング

# pad を config にも反映
for m in (model, ref_model):
    if m is None:
        continue
    try:
        m.config.pad_token_id = tokenizer.pad_token_id
    except:
        pass
    if hasattr(m, "pretrained_model"):
        try:
            m.pretrained_model.config.pad_token_id = tokenizer.pad_token_id
        except:
            pass

ppo_trainer = PPOTrainer(
    args.ppo_config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
)


# =========================
# 報酬モデル（パイプライン）
# =========================
if ppo_trainer.accelerator.num_processes == 1:
    if is_xpu_available():
        pipe_device = "xpu:0"
    elif is_npu_available():
        pipe_device = "npu:0"
    else:
        pipe_device = 0 if torch.cuda.is_available() else "cpu"
else:
    pipe_device = 0 if torch.cuda.is_available() else "cpu"

ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
task, reward_model_name = args.ppo_config.reward_model.split(":")
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline(task, model=reward_model_name, device=pipe_device)
else:
    sentiment_pipe = pipeline(task, model=reward_model_name, device=pipe_device)

if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id
if getattr(sentiment_pipe.model.config, "pad_token_id", None) is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

id2label = sentiment_pipe.model.config.id2label
POS_IDX = [i for i, lab in id2label.items() if str(lab).upper().startswith("POS")][0]
NEG_IDX = [i for i, lab in id2label.items() if str(lab).upper().startswith("NEG")][0]


# =========================
# 生成設定（nucleus-only）
# =========================
output_length_sampler = LengthSampler(args.min_new_tokens, args.max_new_tokens)
generation_kwargs = dict(
    max_new_tokens=args.max_new_tokens,
    do_sample=True,
    top_p=args.top_p,
    top_k=0,  # nucleus-only
    temperature=args.temperature,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)


# =========================
# 学習ループ（最大100ステップ）
# =========================
MAX_STEPS = int(args.max_steps)
for step, batch in tqdm(enumerate(ppo_trainer.dataloader), desc="PPO steps"):
    if step >= MAX_STEPS:
        break

    query_tensors = batch["input_ids"]
    generation_kwargs["max_new_tokens"] = output_length_sampler()

    # プロンプト込みで生成
    response_tensors = ppo_trainer.generate(
        query_tensors,
        return_prompt=True,
        generate_ref_response=False,
        **generation_kwargs
    )

    # ------ response_masks を明示（生成トークンのみ 1） ------
    response_masks = []
    for q, r in zip(query_tensors, response_tensors):
        q_len = int(q.shape[-1])
        r_len = int(r.shape[-1])
        mask = torch.zeros_like(r, dtype=torch.bool, device=r.device)
        mask[q_len:r_len] = True
        response_masks.append(mask)
    # --------------------------------------------------------

    # 可視化用
    responses = ppo_trainer.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    batch["response"] = responses

    # ===== 報酬 =====
    pipe_outputs = sentiment_pipe(responses, **SENT_KW)
    device = ppo_trainer.accelerator.device
    pos = torch.stack([torch.tensor(o[POS_IDX]["score"], dtype=torch.float32, device=device) for o in pipe_outputs])
    neg = torch.stack([torch.tensor(o[NEG_IDX]["score"], dtype=torch.float32, device=device) for o in pipe_outputs])

    if args.reward_mode.lower() == "diff":
        # 差分報酬（-scale..+scale）
        raw_rewards_tensor = (pos - neg) * float(args.reward_scale)
    else:
        # 既定：POS 確率のみ（常に 0..scale の正側）
        raw_rewards_tensor = pos * float(args.reward_scale)

    # ログ
    wandb.log({
        "env/raw_reward_mean": raw_rewards_tensor.mean().item(),
        "env/raw_reward_std":  raw_rewards_tensor.std(unbiased=False).item(),
        "env/raw_reward_min":  raw_rewards_tensor.min().item(),
        "env/raw_reward_max":  raw_rewards_tensor.max().item(),
        "env/reward_scale": float(args.reward_scale),
        "env/reward_mode_is_diff": int(args.reward_mode.lower() == "diff"),
    })

    # PPO に渡す（list[Tensor]）
    rewards = [r.unsqueeze(0).detach() for r in raw_rewards_tensor]

    # PPO step（マスク渡す）
    stats = ppo_trainer.step(
        query_tensors,
        response_tensors,
        rewards,
        response_masks=response_masks
    )

    # 不要キー掃除
    try:
        stats.pop("temp/average_new_tokens")
    except KeyError:
        pass

    # 重要メトリクス
    for k in ("objective/kl", "ppo/ratio", "ppo/policy/entropy"):
        if k in stats:
            wandb.log({k: stats[k]})

    # テキストもログ（重ければ columns を減らす）
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response"])

print("Training finished up to", MAX_STEPS, "steps.")

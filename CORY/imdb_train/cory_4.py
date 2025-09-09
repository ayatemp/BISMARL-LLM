# coding: utf-8
# Dual-Role PPO with LoRA on IMDB (stable generation; no min_length)

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import tyro
import wandb
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    set_seed,
)
from trl.core import LengthSampler


def to_pos_scores(outputs):
    """Robustly extract POSITIVE score from pipeline outputs."""
    scores = []
    for item in outputs:
        if isinstance(item, list):
            pos = None
            for d in item:
                if "POS" in d["label"].upper():
                    pos = d["score"]
                    break
            if pos is None:
                pos = max(x["score"] for x in item)
        else:
            pos = item["score"] if "POS" in item["label"].upper() else (1.0 - item["score"])
        scores.append(torch.tensor(float(pos), dtype=torch.float32))
    return scores



def moving_avg(xs, k=20):
    out, s, q = [], 0.0, []
    for v in xs:
        q.append(v); s += v
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


@dataclass
class ScriptArguments:
    model_name: str = "gpt2-medium"
    query_dataset: str = "imdb"
    reward_model: str = "sentiment-analysis:lvwerra/distilbert-imdb"

    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="gpt2-medium",
            log_with="wandb",
            tracker_project_name=os.environ.get("WANDB_PROJECT", "imdb-ppo"),
            tracker_kwargs={"wandb": {"entity": os.environ.get("WANDB_ENTITY", None)}},
            seed=123,
            learning_rate=1.5e-5,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=1,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,
            kl_penalty="kl",
            init_kl_coef=0.2,
            target_kl=0.05,
            adap_kl_ctrl=True,
            max_grad_norm=1.0,
        )
    )

    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"],
        )
    )

    # generation (no min_length)
    min_new_tokens: int = 8
    max_new_tokens: int = 16
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.05

    # training length
    total_steps: int = 1200
    swap_every: int = 5

    group: Optional[str] = "imdb-dual-role-ppo-lora-stable"
    run_name: Optional[str] = f"dual-role-stable-{int(time.time())}"


args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print(">> Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])


def build_dataset(model_name: str, query_dataset: str, min_len: int = 40, max_prompt_tokens: int = 30):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    ds = load_dataset(query_dataset, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > min_len)

    def to_query(sample):
        ids = tok.encode(sample["review"])[: max_prompt_tokens]
        sample["input_ids"] = ids
        sample["query"] = tok.decode(ids)
        return sample

    ds = ds.map(to_query, batched=False)
    return ds


dataset = build_dataset(args.model_name, args.query_dataset)


def collate(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    queries = [f["query"] for f in features]
    return {"input_ids": input_ids, "query": queries}


model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.model_name,
    peft_config=args.peft_config,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id

ppo_trainer = PPOTrainer(
    args.ppo_config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collate,
)

device = ppo_trainer.accelerator.device

task, reward_model_name = args.reward_model.split(":")
sentiment = pipeline(
    task,
    model=reward_model_name,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
    function_to_apply="none",
    batch_size=16,
)
if sentiment.tokenizer.pad_token_id is None:
    sentiment.tokenizer.pad_token_id = tokenizer.pad_token_id
if sentiment.model.config.pad_token_id is None:
    sentiment.model.config.pad_token_id = tokenizer.pad_token_id

gen_kwargs_base = dict(
    do_sample=True,
    top_k=args.top_k,
    top_p=args.top_p,
    temperature=args.temperature,
    repetition_penalty=args.repetition_penalty,
    min_new_tokens=args.min_new_tokens,
    max_new_tokens=args.max_new_tokens,
    pad_token_id=tokenizer.eos_token_id,
)

merge_template = 'Please rewrite this to sound more positive while keeping meaning: "{}"'
len_sampler = LengthSampler(8, 16)

wandb.init(
    project=args.ppo_config.tracker_project_name,
    group=args.group,
    name=args.run_name,
    reinit=True,
)

print(">> Training start (LoRA dual-role single-policy, stabilized).")

role_flag = False
reward_hist: List[float] = []

# for final printout
last_obs_text: List[str] = []
last_pio_text: List[str] = []

for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=args.total_steps, desc="Dual-Role PPO (LoRA)"):
    if step >= args.total_steps:
        break

    q_ids: List[torch.Tensor] = batch["input_ids"]
    q_txt: List[str] = batch["query"]

    gen_kwargs = dict(gen_kwargs_base)
    mn = int(max(4, len_sampler()))
    gen_kwargs["min_new_tokens"] = mn
    gen_kwargs["max_new_tokens"] = max(mn, args.max_new_tokens)

    # role A: observer
    resp_obs_t = ppo_trainer.generate(q_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    resp_obs = tokenizer.batch_decode(resp_obs_t, skip_special_tokens=True)

    # role B: pioneer
    merged_prompts = [merge_template.format(q + r) for q, r in zip(q_txt, resp_obs)]
    merged_ids = [torch.tensor(tokenizer.encode(mp), dtype=torch.long) for mp in merged_prompts]
    resp_pio_t = ppo_trainer.generate(merged_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    resp_pio = tokenizer.batch_decode(resp_pio_t, skip_special_tokens=True)

    # keep last texts for printing at the end
    last_obs_text = resp_obs
    last_pio_text = resp_pio

    # rewards
    out_obs = sentiment([q + r for q, r in zip(q_txt, resp_obs)])
    out_pio = sentiment([q + r for q, r in zip(q_txt, resp_pio)])
    pos_obs = to_pos_scores(out_obs)
    pos_pio = to_pos_scores(out_pio)

    # alternate stepping
    if not role_flag:
        stats = ppo_trainer.step(q_ids, resp_obs_t, pos_obs)
        stats2 = ppo_trainer.step(merged_ids, resp_pio_t, pos_pio)
    else:
        stats = ppo_trainer.step(merged_ids, resp_pio_t, pos_pio)
        stats2 = ppo_trainer.step(q_ids, resp_obs_t, pos_obs)

    mean_reward = float(torch.stack(pos_obs + pos_pio).mean().item())
    reward_hist.append(mean_reward)
    smooth = moving_avg(reward_hist, 20)[-1]
    wandb.log({
        "env/reward_mean": mean_reward,
        "env/reward_smooth": smooth,
        "ppo/kl_coef": stats.get("ppo/kl_coef", 0.0),
        "ppo/mean_non_score_reward": stats.get("ppo/mean_non_score_reward", 0.0),
        "env/role_flag": int(role_flag),
        "trainer/step": step,
    })

    if (step + 1) % args.swap_every == 0:
        role_flag = not role_flag

print(">> Training finished.")
if last_obs_text:
    print("OBS sample:", last_obs_text[0][:200])
if last_pio_text:
    print("PIO sample:", last_pio_text[0][:200])

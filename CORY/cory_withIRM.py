# coding: utf-8
# Dual-Role PPO with LoRA on IMDB
#  - Reward: IRM raw score (with scalable normalization)
#  - Stable generation; no min_length constraint for prompts

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import tyro
import wandb
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from trl import (
    PPOConfig,
    PPOTrainer,
    AutoModelForCausalLMWithValueHead,
    set_seed,
)
from trl.core import LengthSampler


# =========================
# IRM reward (lightweight)
# =========================
class IRMReward:
    """
    Load a regression head from irm_model_dir and return:
      - raw:    ~[1,10] regression output
      - norm01: (raw-1)/9 clipped to [0,1]
    No SciPy / no Trainer import chain.
    """
    def __init__(
        self,
        irm_model_dir: str,
        max_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.tok = AutoTokenizer.from_pretrained(irm_model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(irm_model_dir)
        self.max_length = max_length
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype or torch.float32
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, idea_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          raw   : (B,) float tensor, IRM regression (~1..10)
          norm01: (B,) float tensor, linearly normalized to [0,1]
        """
        batch = self.tok(
            idea_texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.model(**batch).logits.squeeze(-1).detach().float()  # (B,)
        raw = logits
        norm01 = torch.clamp((raw - 1.0) / 9.0, 0.0, 1.0)
        return raw, norm01


# ==============
# Small utils
# ==============
def moving_avg(xs, k=20):
    out, s, q = [], 0.0, []
    for v in xs:
        q.append(float(v)); s += float(v)
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


class OnlineStats:
    """Welford's algorithm for running mean/std."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: torch.Tensor):
        x = x.detach().float().cpu()
        for v in x:
            self.n += 1
            delta = float(v) - self.mean
            self.mean += delta / self.n
            self.M2 += delta * (float(v) - self.mean)

    @property
    def var(self):
        return (self.M2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def std(self):
        return self.var ** 0.5


# ======================
# Arguments / Defaults
# ======================
@dataclass
class ScriptArguments:
    # policy / data
    model_name: str = "gpt2-medium"
    query_dataset: str = "imdb"

    # IRM reward（CORY/ から見て IRM/irm_iclr_model を既定パスに）
    irm_model_dir: str = "../IRM/irm_iclr_model"
    irm_max_len: int = 512

    # reward scaling
    # - identity  : use norm01 as-is [0,1]
    # - affine    : (norm01-0.5)*beta + 0.5
    # - zscore    : zscore(norm01)*beta + 0.5
    # - raw_identity : use raw (~1..10) directly
    # - raw_affine   : (raw-5.5)*beta + 0.5
    # - raw_zscore   : zscore(raw)*beta (recommended default)
    reward_scale_mode: str = "raw_zscore"
    reward_scale_beta: float = 0.5
    reward_clip_min: float = -4.0
    reward_clip_max: float = 4.0

    # PPO config (slightly looser KL than sentiment setting)
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
            init_kl_coef=0.05,   # looser than 0.2
            target_kl=0.08,      # slightly higher target
            adap_kl_ctrl=True,
            max_grad_norm=1.0,
        )
    )

    # LoRA
    peft_config: Optional[LoraConfig] = field(
        default_factory=lambda: LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "c_fc"],
        )
    )

    # generation (longer so IRM can judge)
    min_new_tokens: int = 12
    max_new_tokens: int = 24
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.05

    # training length
    total_steps: int = 3000
    swap_every: int = 5

    # logging
    group: Optional[str] = "imdb-dual-role-ppo-lora-irm"
    run_name: Optional[str] = f"dual-role-irm-{int(time.time())}"


# =========
# Main
# =========
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


# policy / tokenizer
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

# IRM reward instance
irm_reward = IRMReward(args.irm_model_dir, max_length=args.irm_max_len, device=device)

# online stats for zscore modes
online_stats_raw = OnlineStats()
online_stats_n01 = OnlineStats()


def scale_rewards(raw: torch.Tensor, n01: torch.Tensor) -> List[torch.Tensor]:
    """
    raw: (B,)  ~1..10
    n01: (B,)  in [0,1]
    returns: List[Tensor(float32)] on device for PPO
    """
    mode = args.reward_scale_mode
    beta = float(args.reward_scale_beta)

    if mode == "identity":
        r = n01.float()

    elif mode == "affine":
        r = (n01.float() - 0.5) * beta + 0.5

    elif mode == "zscore":
        online_stats_n01.update(n01)
        mu, std = online_stats_n01.mean, max(online_stats_n01.std, 1e-6)
        r = (n01.float() - mu) / std * beta + 0.5

    elif mode == "raw_identity":
        r = raw.float()

    elif mode == "raw_affine":
        # center around 5.5 (mid of 1..10), then squeeze by beta and shift to ~0.5 scale
        r = (raw.float() - 5.5) * beta + 0.5

    elif mode == "raw_zscore":
        online_stats_raw.update(raw)
        mu, std = online_stats_raw.mean, max(online_stats_raw.std, 1e-6)
        r = (raw.float() - mu) / std * beta
        # Note: no +0.5 shift needed; PPO can learn on centered rewards as well.

    else:
        r = n01.float()

    r = torch.clamp(r, min=float(args.reward_clip_min), max=float(args.reward_clip_max))
    return [t.detach().to(torch.float32).to(device) for t in r]


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

# 2-agent interaction template（必要に応じて調整）
merge_template = 'Please rewrite this to sound more positive while keeping meaning: "{}"'
len_sampler = LengthSampler(8, 16)  # 生成のバラし。min/maxは上で最終的に決める。

wandb.init(
    project=args.ppo_config.tracker_project_name,
    group=args.group,
    name=args.run_name,
    reinit=True,
)

print(">> Training start (LoRA dual-role with IRM RAW reward).")

role_flag = False
reward_hist: List[float] = []

# for final printout
last_obs_text: List[str] = []
last_pio_text: List[str] = []

for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=args.total_steps, desc="Dual-Role PPO (LoRA+IRM)"):
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

    # ---- IRM rewards on ideas (q + r) ----
    idea_obs = [q + r for q, r in zip(q_txt, resp_obs)]
    idea_pio = [q + r for q, r in zip(q_txt, resp_pio)]

    raw_obs_t, n01_obs_t = irm_reward.score(idea_obs)   # (B,), (B,)
    raw_pio_t, n01_pio_t = irm_reward.score(idea_pio)

    pos_obs = scale_rewards(raw_obs_t, n01_obs_t)  # List[Tensor]
    pos_pio = scale_rewards(raw_pio_t, n01_pio_t)

    # ===== PPO step (alternate) =====
    if not role_flag:
        stats = ppo_trainer.step(q_ids, resp_obs_t, pos_obs)
        stats2 = ppo_trainer.step(merged_ids, resp_pio_t, pos_pio)
    else:
        stats = ppo_trainer.step(merged_ids, resp_pio_t, pos_pio)
        stats2 = ppo_trainer.step(q_ids, resp_obs_t, pos_obs)

    # ===== Logging =====
    with torch.no_grad():
        raw_all    = torch.cat([raw_obs_t, raw_pio_t]).detach().float().cpu()
        n01_all    = torch.cat([n01_obs_t, n01_pio_t]).detach().float().cpu()
        scaled_all = torch.stack(pos_obs + pos_pio).detach().float().cpu()

        # histogram/quantiles/clip-rate of scaled rewards
        clip_min, clip_max = args.reward_clip_min, args.reward_clip_max
        clip_mask = (scaled_all <= clip_min + 1e-6) | (scaled_all >= clip_max - 1e-6)
        clip_frac = float(clip_mask.float().mean().item())

        # response length
        resp_len_obs = float(torch.tensor([len(x.split()) for x in resp_obs]).float().mean().item())
        resp_len_pio = float(torch.tensor([len(x.split()) for x in resp_pio]).float().mean().item())

    reward_hist.append(float(scaled_all.mean().item()))
    smooth = moving_avg(reward_hist, 20)[-1]

    # base env logs
    wandb.log({
        "env/reward_raw_mean":  float(raw_all.mean().item()),
        "env/reward_raw_std":   float(raw_all.std().item()),
        "env/reward_mean_scaled": float(scaled_all.mean().item()),
        "env/reward_std_scaled":  float(scaled_all.std().item()),
        "env/reward_smooth_scaled": smooth,
        "env/reward_scaled_clip_frac": clip_frac,
        "env/reward_scaled_p05": float(torch.quantile(scaled_all, 0.05).item()),
        "env/reward_scaled_p25": float(torch.quantile(scaled_all, 0.25).item()),
        "env/reward_scaled_p50": float(torch.quantile(scaled_all, 0.50).item()),
        "env/reward_scaled_p75": float(torch.quantile(scaled_all, 0.75).item()),
        "env/reward_scaled_p95": float(torch.quantile(scaled_all, 0.95).item()),
        "generation/resp_len_mean": (resp_len_obs + resp_len_pio) / 2.0,
        "generation/resp_len_obs": resp_len_obs,
        "generation/resp_len_pio": resp_len_pio,
        "env/role_flag": int(role_flag),
        "trainer/step": step,
        "env/reward_scaled_hist": wandb.Histogram(scaled_all.numpy()),
    })

    # mirror PPO stats
    wandb.log({f"ppo/{k}": v for k, v in stats.items()  if isinstance(v, (int, float))})
    wandb.log({f"ppo_b/{k}": v for k, v in stats2.items() if isinstance(v, (int, float))})

    if (step + 1) % args.swap_every == 0:
        role_flag = not role_flag

print(">> Training finished.")
if last_obs_text:
    print("OBS sample:", last_obs_text[0][:200])
if last_pio_text:
    print("PIO sample:", last_pio_text[0][:200])

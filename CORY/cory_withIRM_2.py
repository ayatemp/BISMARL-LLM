# coding: utf-8
# Dual-Role PPO with LoRA on IMDB
# Reward: IRM raw score (EMA zscore + tanh-squash, safe & stable)

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    Load IRM regression head and return:
      - raw   ~ [1,10] (regression output)
      - norm01 = clip((raw-1)/9, 0, 1)
    """
    def __init__(self, irm_model_dir: str, max_length: int = 512, device=None, dtype=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(irm_model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(irm_model_dir)
        self.max_length = max_length
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype or torch.float32
        self.model.to(self.device).eval()

    @torch.no_grad()
    def score(self, idea_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
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


class EMAStats:
    """EMA-based running mean/std (より安定)"""
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._mean = None
        self._var = None
        self._eps = 1e-6

    def update(self, x: torch.Tensor):
        x = x.detach().float().mean()
        if self._mean is None:
            self._mean = x
            self._var = torch.tensor(1.0, device=x.device)  # 初期分散は1で開始
        else:
            self._mean = self.decay * self._mean + (1 - self.decay) * x
            # 分散のEMA（スカラーで十分）
            self._var = self.decay * self._var + (1 - self.decay) * (x - self._mean).pow(2)

    @property
    def mean(self):
        return float(self._mean.item() if self._mean is not None else 0.0)

    @property
    def std(self):
        if self._var is None:
            return 1.0
        return float(torch.sqrt(self._var + 1e-6).item())


# ======================
# Arguments / Defaults
# ======================
@dataclass
class ScriptArguments:
    # policy / data
    model_name: str = "gpt2-medium"
    query_dataset: str = "imdb"

    # IRM reward（CORY/ から見て IRM/irm_iclr_model が既定）
    irm_model_dir: str = "../IRM/irm_iclr_model"
    irm_max_len: int = 512

    # reward scaling
    #   - raw_zscore_tanh: EMA zscore -> beta -> tanh squash (推奨)
    #   - raw_identity   : rawそのまま
    #   - zscore         : norm01のzscore(+0.5) [参考/互換]
    reward_scale_mode: str = "raw_zscore_tanh"
    reward_scale_beta: float = 0.15      # 小さく開始し、伸びが鈍ければ徐々に↑
    reward_clip_min: float = -2.0        # 二次防御（tanh前後で丸めるので広めでOK）
    reward_clip_max: float =  2.0

    # PPO config（KLをやや強め、ratio暴走を抑止）
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
            ratio_threshold=5.0,  # ← 平均ratioが跳ねるバッチを早めにskip
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

    # generation（IRMが判断しやすい長さに）
    min_new_tokens: int = 12
    max_new_tokens: int = 32
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

# 軽い禁止トークン（置換文字など）で文字化けモードを抑止
BAD_WORDS = ["�"]  # 必要に応じて追加
bad_words_ids = tokenizer([*BAD_WORDS], add_special_tokens=False).input_ids

ppo_trainer = PPOTrainer(
    args.ppo_config,
    model,
    ref_model=None,  # refを与えない運用（KLは内部近似）。必要なら初期重みでrefを別途作る
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collate,
)

device = ppo_trainer.accelerator.device

# IRM reward instance
irm_reward = IRMReward(args.irm_model_dir, max_length=args.irm_max_len, device=device)

# EMA統計（raw / norm01 それぞれ）
ema_raw = EMAStats(decay=0.99)
ema_n01 = EMAStats(decay=0.99)


def scale_rewards(raw: torch.Tensor, n01: torch.Tensor) -> List[torch.Tensor]:
    """
    raw: (B,)  ~1..10
    n01: (B,)  in [0,1]
    returns: List[Tensor(float32)] on device for PPO
    """
    mode = args.reward_scale_mode
    beta = float(args.reward_scale_beta)

    if mode == "raw_identity":
        r = raw.float()

    elif mode == "raw_zscore_tanh":
        # EMA zscore -> beta -> tanh squash（推奨）
        ema_raw.update(raw)
        mu, std = ema_raw.mean, max(ema_raw.std, 1e-6)
        z = (raw.float() - mu) / std
        r = torch.tanh(z * beta)  # [-1,1]に自然丸め

    elif mode == "zscore":
        ema_n01.update(n01)
        mu, std = ema_n01.mean, max(ema_n01.std, 1e-6)
        r = (n01.float() - mu) / std * beta + 0.5  # 互換運用

    else:  # 既定以外は素直に norm01 を返す
        r = n01.float()

    # 二次防御クリップ
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
    bad_words_ids=bad_words_ids if bad_words_ids else None,
)

# 2-agent interaction template（必要に応じて調整）
merge_template = 'Please rewrite this to sound more positive while keeping meaning: "{}"'
len_sampler = LengthSampler(8, 16)  # 生成のバラし

wandb.init(
    project=args.ppo_config.tracker_project_name,
    group=args.group,
    name=args.run_name,
    reinit=True,
)

print(">> Training start (LoRA dual-role with IRM RAW reward, safe scaling).")

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

    r_obs = scale_rewards(raw_obs_t, n01_obs_t)  # List[Tensor]
    r_pio = scale_rewards(raw_pio_t, n01_pio_t)

    # ===== PPO step (alternate) =====
    if not role_flag:
        stats = ppo_trainer.step(q_ids, resp_obs_t, r_obs)
        stats2 = ppo_trainer.step(merged_ids, resp_pio_t, r_pio)
    else:
        stats = ppo_trainer.step(merged_ids, resp_pio_t, r_pio)
        stats2 = ppo_trainer.step(q_ids, resp_obs_t, r_obs)

    # ===== Logging =====
    with torch.no_grad():
        raw_all    = torch.cat([raw_obs_t, raw_pio_t]).detach().float().cpu()
        n01_all    = torch.cat([n01_obs_t, n01_pio_t]).detach().float().cpu()
        scaled_all = torch.stack(r_obs + r_pio).detach().float().cpu()

        clip_min, clip_max = args.reward_clip_min, args.reward_clip_max
        clip_mask = (scaled_all <= clip_min + 1e-6) | (scaled_all >= clip_max - 1e-6)
        clip_frac = float(clip_mask.float().mean().item())

        resp_len_obs = float(torch.tensor([len(x.split()) for x in resp_obs]).float().mean().item())
        resp_len_pio = float(torch.tensor([len(x.split()) for x in resp_pio]).float().mean().item())

    reward_hist.append(float(scaled_all.mean().item()))
    smooth = moving_avg(reward_hist, 20)[-1]

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
    wandb.log({f"ppo/{k}": v for k, v in stats.items()  if isinstance(v, (int, float))})
    wandb.log({f"ppo_b/{k}": v for k, v in stats2.items() if isinstance(v, (int, float))})

    if (step + 1) % args.swap_every == 0:
        role_flag = not role_flag

print(">> Training finished.")
if last_obs_text:
    print("OBS sample:", last_obs_text[0][:200])
if last_pio_text:
    print("PIO sample:", last_pio_text[0][:200])

ppo_trainer.save_pretrained("./outputs/cory_withIRM_last")
tokenizer.save_pretrained("./outputs/cory_withIRM_last")
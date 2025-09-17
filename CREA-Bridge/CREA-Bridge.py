# coding: utf-8
"""
CREA-Bridge: Pseudo-Multi-Agent PPO with IRM + Bridger Band-Pass Reward
- Roles: Bridger -> Observer -> Pioneer (single-LLM, role-conditioned prompts)
- Rewards:
    * IRM (Idea Reward Model) on the final Pioneer idea (0..1 scaled, or raw with z-tanh)
    * Bridger band-pass reward: R_bridge = g(sim(ϕ(x), ϕ(y_br)); τ_min, τ_max)
      where sim = cosine_similarity on encoder embeddings.
- Training: TRL PPO with LoRA (optional), KL control, safe reward scaling

Run (example):
  conda activate cory
  python CREA-Bridge.py \
    --model_name gpt2-medium \
    --irm_model_dir ../IRM/irm_iclr_model \
    --embed_model sentence-transformers/all-MiniLM-L6-v2 \
    --query_dataset imdb \
    --total_steps 3000 \
    --tau_min 0.25 --tau_max 0.55 --bridge_weight 0.5

Notes:
- Defaults are chosen to work in the "cory" env used by cory_withIRM_2.py.
- If the embedding model is unavailable, set --embed_model to a lightweight encoder
  available in your environment (e.g., "intfloat/e5-small-v2" or "BAAI/bge-small-en-v1.5").
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn.functional as F
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
)
from transformers import set_seed
from trl.core import LengthSampler


# ============================================================
# IRM reward (shared with CORY-withIRM)
# ============================================================
class IRMReward:
    """
    Lightweight IRM loader
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


# ============================================================
# EMA stats for reward stabilization
# ============================================================
class EMAStats:
    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._mean = None
        self._var = None

    def update(self, x: torch.Tensor):
        x = x.detach().float().mean()
        if self._mean is None:
            self._mean = x
            self._var = torch.tensor(1.0, device=x.device)
        else:
            self._mean = self.decay * self._mean + (1 - self.decay) * x
            self._var = self.decay * self._var + (1 - self.decay) * (x - self._mean).pow(2)

    @property
    def mean(self):
        return float(self._mean.item() if self._mean is not None else 0.0)

    @property
    def std(self):
        import math
        if self._var is None:
            return 1.0
        return float(torch.sqrt(self._var + 1e-6).item())


# ============================================================
# Bridger: embedding model + band-pass reward
# ============================================================
class TextEmbedder:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        batch = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        # simple pooling: mean over tokens (works across many encoders)
        if hasattr(out, "last_hidden_state"):
            x = out.last_hidden_state  # (B, T, D)
            mask = batch["attention_mask"].unsqueeze(-1)  # (B, T, 1)
            x = (x * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-6)
            x = F.normalize(x, p=2, dim=-1)
            return x
        raise RuntimeError("Encoder does not expose last_hidden_state")


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    return (a * b).sum(dim=-1)


def bandpass_reward(sim: torch.Tensor, tau_min: float, tau_max: float, sharpness: float = 12.0) -> torch.Tensor:
    """
    Smooth band-pass shape that peaks in [tau_min, tau_max] and decays outside.
    We use a product of two logistic CDFs to form a "flat-ish" top with soft edges.
    R = σ(sharp*(sim - tau_min)) * σ(sharp*(tau_max - sim))
    Then rescale to ~[0,1].
    """
    s = sim.clamp(-1.0, 1.0)
    left = torch.sigmoid(sharpness * (s - tau_min))
    right = torch.sigmoid(sharpness * (tau_max - s))
    r = left * right  # peak in between
    # normalize to [0,1] roughly (max≈0.25 when symmetric); scale by 4.
    r = torch.clamp(4.0 * r, 0.0, 1.0)
    return r


# ============================================================
# Arguments / Defaults
# ============================================================
@dataclass
class ScriptArguments:
    # policy / data
    model_name: str = "gpt2-medium"
    query_dataset: str = "imdb"

    # IRM
    irm_model_dir: str = "../IRM/irm_iclr_model"
    irm_max_len: int = 512

    # Bridger encoder
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tau_min: float = 0.25
    tau_max: float = 0.55
    bridge_weight: float = 0.5   # weight for Bridger reward when mixing
    bridge_steps: int = 1        # number of PPO steps applied to Bridger per outer step

    # reward scaling
    reward_scale_mode: str = "raw_zscore_tanh"  # for IRM
    reward_scale_beta: float = 0.15
    reward_clip_min: float = -2.0
    reward_clip_max: float = 2.0

    # PPO config
    ppo_config: PPOConfig = field(
        default_factory=lambda: PPOConfig(
            model_name="gpt2-medium",
            log_with="wandb",
            tracker_project_name=os.environ.get("WANDB_PROJECT", "crea-bridge"),
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
            ratio_threshold=5.0,
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

    # generation
    min_new_tokens: int = 12
    max_new_tokens: int = 48
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0
    repetition_penalty: float = 1.05

    # training length
    total_steps: int = 3000
    swap_every: int = 5  # swap Pioneer/Observer update order

    # logging
    group: Optional[str] = "crea-bridge"
    run_name: Optional[str] = f"crea-bridge-{int(time.time())}"


# ============================================================
# Utilities
# ============================================================

def moving_avg(xs, k=20):
    out, s, q = [], 0.0, []
    for v in xs:
        q.append(float(v)); s += float(v)
        if len(q) > k:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def build_dataset(model_name: str, query_dataset: str, min_len: int = 40, max_prompt_tokens: int = 30):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    ds = load_dataset(query_dataset, split="train")
    # imdb: field name is "text"; make it robust via get
    if "text" in ds.column_names:
        src_col = "text"
    elif "review" in ds.column_names:
        src_col = "review"
    else:
        src_col = ds.column_names[0]
    ds = ds.rename_columns({src_col: "review"})
    ds = ds.filter(lambda x: len(x["review"]) > min_len)

    def to_query(sample):
        ids = tok.encode(sample["review"])[: max_prompt_tokens]
        sample["input_ids"] = ids
        sample["query"] = tok.decode(ids)
        return sample

    ds = ds.map(to_query, batched=False)
    return ds


# ============================================================
# Main
# ============================================================
args = tyro.cli(ScriptArguments)
set_seed(args.ppo_config.seed)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
print(">> Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])  # informative only

# Dataset
dataset = build_dataset(args.model_name, args.query_dataset)

def collate(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    queries = [f["query"] for f in features]
    return {"input_ids": input_ids, "query": queries}

# Policy / tokenizer
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    args.model_name,
    peft_config=args.peft_config,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Keep outputs sane
BAD_WORDS = ["�"]
bad_words_ids = tokenizer([*BAD_WORDS], add_special_tokens=False).input_ids

ppo_trainer = PPOTrainer(
    args.ppo_config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collate,
)

device = ppo_trainer.accelerator.device

# IRM and Bridger
irm_reward = IRMReward(args.irm_model_dir, max_length=args.irm_max_len, device=device)
embedder = TextEmbedder(args.embed_model, device=device)
ema_raw = EMAStats(decay=0.99)
ema_n01 = EMAStats(decay=0.99)


def scale_rewards(raw: torch.Tensor, n01: torch.Tensor) -> List[torch.Tensor]:
    mode = args.reward_scale_mode
    beta = float(args.reward_scale_beta)

    if mode == "raw_identity":
        r = raw.float()
    elif mode == "raw_zscore_tanh":
        ema_raw.update(raw)
        mu, std = ema_raw.mean, max(ema_raw.std, 1e-6)
        z = (raw.float() - mu) / std
        r = torch.tanh(z * beta)
    elif mode == "zscore":
        ema_n01.update(n01)
        mu, std = ema_n01.mean, max(ema_n01.std, 1e-6)
        r = (n01.float() - mu) / std * beta + 0.5
    else:
        r = n01.float()

    r = torch.clamp(r, min=float(args.reward_clip_min), max=float(args.reward_clip_max))
    return [t.detach().to(torch.float32).to(device) for t in r]


# Generation kwargs
len_sampler = LengthSampler(8, 16)

gen_kwargs_base = dict(
    do_sample=True,
    top_k=args.top_k,
    top_p=args.top_p,
    temperature=args.temperature,
    repetition_penalty=args.repetition_penalty,
    min_new_tokens=args.min_new_tokens,
    max_new_tokens=args.max_new_tokens,
    pad_token_id=tokenizer.pad_token_id,
    bad_words_ids=bad_words_ids if bad_words_ids else None,
)

# Role prompts (concise, role-conditioned)
bridger_prompt = (
    "You are the Bridger. Propose a short, distant-yet-relevant concept from another domain "
    "that could inspire a novel angle. Keep it 1-2 sentences. Query: \"{q}\"\nBridger: "
)

observer_prompt = (
    "You are the Observer. Given the user query and the Bridger hint, critique or refine the idea "
    "focusing on usefulness, clarity, and feasibility in 1-2 sentences.\nQuery: \"{q}\"\nBridger hint: {b}\nObserver: "
)

pioneer_prompt = (
    "You are the Pioneer. Using the Observer feedback and Bridger hint, produce a concrete, positive "
    "proposal (2-3 sentences) that remains faithful to the query while adding a creative twist.\n"
    "Query: \"{q}\"\nBridger hint: {b}\nObserver: {o}\nPioneer: "
)

wandb.init(
    project=args.ppo_config.tracker_project_name,
    group=args.group,
    name=args.run_name,
    reinit=True,
)

print(">> Training CREA-Bridge (LoRA + PPO + IRM + Bridger band-pass)")

role_flag = False  # swap observer/pioneer update order periodically
reward_hist: List[float] = []

last_b_text: List[str] = []
last_o_text: List[str] = []
last_p_text: List[str] = []

for step, batch in tqdm(enumerate(ppo_trainer.dataloader), total=args.total_steps, desc="CREA-Bridge PPO"):
    if step >= args.total_steps:
        break

    q_ids: List[torch.Tensor] = batch["input_ids"]
    q_txt: List[str] = batch["query"]

    gen_kwargs = dict(gen_kwargs_base)
    mn = int(max(4, len_sampler()))
    gen_kwargs["min_new_tokens"] = mn
    gen_kwargs["max_new_tokens"] = max(mn, args.max_new_tokens)

    # --------------------------------------------------
    # (1) Bridger: generate cross-domain hint
    # --------------------------------------------------
    bridger_inputs = [bridger_prompt.format(q=q) for q in q_txt]
    bridger_ids = [torch.tensor(tokenizer.encode(x), dtype=torch.long) for x in bridger_inputs]

    resp_b_t = ppo_trainer.generate(bridger_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    resp_b = tokenizer.batch_decode(resp_b_t, skip_special_tokens=True)
    last_b_text = resp_b

    # Bridger reward: band-pass similarity
    with torch.no_grad():
        emb_q = embedder.encode(q_txt)
        emb_b = embedder.encode(resp_b)
        sim_b = cosine_sim(emb_q, emb_b)               # (B,)
        r_bridge_01 = bandpass_reward(sim_b, args.tau_min, args.tau_max)  # [0,1]
        # map to [-1,1] around 0.5 (centered) and scale
        r_bridge = (r_bridge_01 - 0.5) * 2.0 * args.bridge_weight
    r_bridge_list = [t.detach().to(torch.float32).to(device) for t in r_bridge]

    # Optionally train Bridger directly (encourage hitting the band)
    for _ in range(max(1, args.bridge_steps)):
        _ = ppo_trainer.step(bridger_ids, resp_b_t, r_bridge_list)

    # --------------------------------------------------
    # (2) Observer: critique using Bridger hint
    # --------------------------------------------------
    observer_inputs = [observer_prompt.format(q=q, b=b) for q, b in zip(q_txt, resp_b)]
    observer_ids = [torch.tensor(tokenizer.encode(x), dtype=torch.long) for x in observer_inputs]

    resp_o_t = ppo_trainer.generate(observer_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    resp_o = tokenizer.batch_decode(resp_o_t, skip_special_tokens=True)
    last_o_text = resp_o

    # --------------------------------------------------
    # (3) Pioneer: produce final proposal
    # --------------------------------------------------
    pioneer_inputs = [pioneer_prompt.format(q=q, b=b, o=o) for q, b, o in zip(q_txt, resp_b, resp_o)]
    pioneer_ids = [torch.tensor(tokenizer.encode(x), dtype=torch.long) for x in pioneer_inputs]

    resp_p_t = ppo_trainer.generate(pioneer_ids, return_prompt=False, generate_ref_response=False, **gen_kwargs)
    resp_p = tokenizer.batch_decode(resp_p_t, skip_special_tokens=True)
    last_p_text = resp_p

    # --------------------------------------------------
    # Rewards for Observer/Pioneer using IRM on the composed idea
    # idea text options: (q + pioneer), we can also include hint/observer implicitly
    # --------------------------------------------------
    idea_p = [q + "\n" + p for q, p in zip(q_txt, resp_p)]
    idea_o = [q + "\n" + o for q, o in zip(q_txt, resp_o)]

    raw_p, n01_p = irm_reward.score(idea_p)
    raw_o, n01_o = irm_reward.score(idea_o)

    r_p = scale_rewards(raw_p, n01_p)
    r_o = scale_rewards(raw_o, n01_o)

    # Alternate step order for observer / pioneer (stability trick)
    if not role_flag:
        stats_o = ppo_trainer.step(observer_ids, resp_o_t, r_o)
        stats_p = ppo_trainer.step(pioneer_ids, resp_p_t, r_p)
    else:
        stats_p = ppo_trainer.step(pioneer_ids, resp_p_t, r_p)
        stats_o = ppo_trainer.step(observer_ids, resp_o_t, r_o)

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    with torch.no_grad():
        raw_all = torch.cat([raw_o, raw_p]).detach().float().cpu()
        n01_all = torch.cat([n01_o, n01_p]).detach().float().cpu()
        scaled_all = torch.stack(r_o + r_p).detach().float().cpu()
        resp_len_b = float(torch.tensor([len(x.split()) for x in resp_b]).float().mean().item())
        resp_len_o = float(torch.tensor([len(x.split()) for x in resp_o]).float().mean().item())
        resp_len_p = float(torch.tensor([len(x.split()) for x in resp_p]).float().mean().item())

        clip_min, clip_max = args.reward_clip_min, args.reward_clip_max
        clip_mask = (scaled_all <= clip_min + 1e-6) | (scaled_all >= clip_max - 1e-6)
        clip_frac = float(clip_mask.float().mean().item())

    reward_hist.append(float(scaled_all.mean().item()))
    smooth = moving_avg(reward_hist, 20)[-1]

    wandb.log({
        "irm/raw_mean": float(raw_all.mean().item()),
        "irm/raw_std": float(raw_all.std().item()),
        "irm/norm01_mean": float(n01_all.mean().item()),
        "irm/norm01_std": float(n01_all.std().item()),
        "reward/mean_scaled": float(scaled_all.mean().item()),
        "reward/std_scaled": float(scaled_all.std().item()),
        "reward/smooth": smooth,
        "reward/clip_frac": clip_frac,
        "bridger/sim_mean": float(sim_b.mean().item()),
        "bridger/sim_std": float(sim_b.std().item()),
        "bridger/r01_mean": float(r_bridge_01.mean().item()),
        "bridger/r01_std": float(r_bridge_01.std().item()),
        "bridger/resp_len": resp_len_b,
        "generation/resp_len_obs": resp_len_o,
        "generation/resp_len_pio": resp_len_p,
        "env/role_flag": int(role_flag),
        "trainer/step": step,
    })

    # log PPO stats (numeric only)
    if isinstance(stats_o, dict):
        wandb.log({f"ppo_obs/{k}": v for k, v in stats_o.items() if isinstance(v, (int, float))})
    if isinstance(stats_p, dict):
        wandb.log({f"ppo_pio/{k}": v for k, v in stats_p.items() if isinstance(v, (int, float))})

    if (step + 1) % args.swap_every == 0:
        role_flag = not role_flag

print(">> Training finished.")
if last_b_text:
    print("BRIDGER sample:", last_b_text[0][:200])
if last_o_text:
    print("OBSERVER sample:", last_o_text[0][:200])
if last_p_text:
    print("PIONEER sample:", last_p_text[0][:200])

# Save checkpoint
save_dir = "./outputs/crea_bridge_last"
ppo_trainer.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f">> Saved to {save_dir}")

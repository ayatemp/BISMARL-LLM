#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CREA-Bridge (single-file, simple & robust)
- Pipeline: Observer(JSON) -> Pioneer(JSON) with small Bridger hints
- Reward: IRM regression head (optionally sliding-window + calibration) -> scaled (zscore→tanh or calibrated prob)
- Training: TRL PPOTrainer if available; else fall back to NO-PPO (generate+score)
- Orchestration: Accelerate; Deepspeed via Accelerate if --use-deepspeed is passed
- Logging: wandb (disable with --disable-wandb)

Notes:
- Keep dependencies modest; fail-safe fallbacks with explicit debug prints
- HF datasets required for PPO dataset stability
"""

from __future__ import annotations
import os, re, json, time, random, argparse, sys, traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# tqdm (optional)
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kw): return it

# wandb (enabled by default; can be disabled via --disable-wandb)
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False
    class _W:  # dummy
        def init(self, **k): pass
        def log(self, d): pass
    wandb = _W()

# Deepspeed (optional; used via Accelerate)
try:
    import deepspeed  # noqa: F401
    _HAS_DS = True
except Exception:
    _HAS_DS = False

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers.utils import is_bitsandbytes_available
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# TRL (optional)
_HAS_TRL = True
try:
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
except Exception:
    _HAS_TRL = False

# datasets (required for PPO dataset)
from datasets import Dataset as HFDataset

# peft (optional for LoRA)
try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

# ================================================================
# JSON utils (robust but minimal)
# ================================================================
REQUIRED_KEYS = ["input_concepts","new_concepts","bridge_rationale","plan","risks","title","abstract"]
_JSON_TITLE_RE = re.compile(r'"title"\s*:\s*"([^"]+)"', re.S)
_JSON_ABS_RE   = re.compile(r'"abstract"\s*:\s*"([^"]+)"', re.S)

def _strip_nonjson_tail(s: str) -> str:
    if not s: return s
    i, j = s.find('{'), s.rfind('}')
    return s[i:j+1] if (i != -1 and j != -1 and j > i) else s

def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    s1 = _strip_nonjson_tail(s).replace("\r","")
    s1 = re.sub(r',\s*([}\]])', r'\1', s1)
    try:
        return json.loads(s1)
    except Exception:
        return None

def validate_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict): return False
    for k in REQUIRED_KEYS:
        if k not in obj: return False
        v = obj[k]
        if isinstance(v, str) and len(v.strip()) < 5: return False
        if isinstance(v, list) and len(v) == 0: return False
    ic = obj.get("input_concepts", []); nc = obj.get("new_concepts", [])
    if isinstance(ic, list) and isinstance(nc, list):
        if set(map(str.lower, ic)) == set(map(str.lower, nc)): return False
    return True

def make_irm_text_from_idea_relaxed(idea: Dict[str, Any]) -> str:
    if not isinstance(idea, dict): return ""
    t = str(idea.get("title","")).strip()
    a = str(idea.get("abstract","")).strip()
    b = str(idea.get("bridge_rationale","")).strip()
    p = str(idea.get("plan","")).strip()
    parts = []
    if t: parts.append(t)
    if a: parts.append(a)
    if b: parts.append(f"Bridge: {b}")
    if p: parts.append(f"Plan: {p}")
    return "\n\n".join(parts) if parts else ""

def fallback_irm_text_from_raw_response(resp_text: str, seed_problem: str = "") -> str:
    if not resp_text:
        head = seed_problem.strip() or "Generated Idea"
        return f"{head}\n\nEmpty response."
    s = _strip_nonjson_tail(resp_text)
    mt, ma = _JSON_TITLE_RE.search(s), _JSON_ABS_RE.search(s)
    if mt or ma:
        t = mt.group(1).strip() if mt else ""
        a = ma.group(1).strip() if ma else s[:1000]
        return (t + "\n\n" + a).strip()
    body = s.strip()[:1600]
    head = seed_problem.strip() or "Generated Idea"
    return f"{head}\n\n{body}"

# ================================================================
# IRM reward (sliding window + optional calibration)
# ================================================================
def _build_piecewise_calibrator_from_buckets(buckets: List[dict]):
    xs, ys = [], []
    for b in buckets:
        if "pred_mean" in b and "true_mean" in b:
            xs.append(float(b["pred_mean"]))
            ys.append(float(b["true_mean"]))
    if len(xs) < 2:
        return lambda v: float(np.clip(v, 0.0, 1.0))
    xs, ys = np.array(xs), np.array(ys)
    order = np.argsort(xs)
    xs, ys = xs[order], np.maximum.accumulate(ys[order])
    def f(v: float) -> float:
        return float(np.interp(v, xs, ys, left=ys[0], right=ys[-1]))
    return f

class IRMReward:
    def __init__(self, irm_model_dir: str, max_length: int = 512,
                 device: Optional[torch.device] = None,
                 quantization: str = "none", torch_dtype: str = "auto",
                 use_sliding: bool = False, stride_ratio: float = 0.5, agg: str = "mean",
                 calib_path: Optional[str] = None):
        self.max_length = max_length
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tok = AutoTokenizer.from_pretrained(irm_model_dir, use_fast=True)

        self.use_sliding = bool(use_sliding)
        self.stride_ratio = float(max(0.01, min(0.99, stride_ratio)))
        self.agg = (agg or "mean").lower().strip()
        self.calib_fn = None

        # model load
        dtype = None
        if torch_dtype == "float16": dtype = torch.float16
        elif torch_dtype == "bfloat16": dtype = torch.bfloat16
        elif torch_dtype == "float32": dtype = torch.float32

        if quantization in ("8bit","4bit") and is_bitsandbytes_available() and BitsAndBytesConfig is not None:
            m = AutoModelForSequenceClassification.from_pretrained(
                irm_model_dir,
                torch_dtype=(torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else
                             (torch.float16 if torch.cuda.is_available() else torch.float32)),
                low_cpu_mem_usage=True
            )
            m.to(self.device).eval()
            self.model = m
        else:
            kw = {}
            if dtype is not None: kw["torch_dtype"] = dtype
            try:
                m = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, low_cpu_mem_usage=True, **kw)
                m.to(self.device).eval()
                self.model = m
            except RuntimeError:
                try:
                    import gc; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception: pass
                cpu = torch.device("cpu")
                m = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, low_cpu_mem_usage=True, **kw)
                m.to(cpu).eval()
                self.model = m

        if calib_path and os.path.isfile(calib_path):
            try:
                with open(calib_path, "r", encoding="utf-8") as f:
                    js = json.load(f)
                buckets = js.get("calibration_buckets_reward") or js.get("calibration_buckets")
                if isinstance(buckets, list) and buckets:
                    self.calib_fn = _build_piecewise_calibrator_from_buckets(buckets)
            except Exception:
                self.calib_fn = None

    def _windows_of(self, text: str) -> List[str]:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        L = len(ids)
        if L <= self.max_length: return [text]
        stride = max(1, int(self.max_length * (1.0 - self.stride_ratio)))
        chunks = []
        for start in range(0, L, stride):
            end = start + self.max_length
            sub_ids = ids[start:end]
            if not sub_ids: continue
            chunks.append(self.tok.decode(sub_ids, skip_special_tokens=True))
            if end >= L: break
        return chunks

    @torch.no_grad()
    def _score_logits(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok(texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        dev = next(self.model.parameters()).device
        enc = {k:(v.to(dev) if torch.is_tensor(v) else v) for k,v in enc.items()}
        logits = self.model(**enc).logits.squeeze(-1).detach().float()
        return logits

    @torch.no_grad()
    def score(self, idea_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if not self.use_sliding:
            raw = self._score_logits(idea_texts)
        else:
            all_chunks, owners = [], []
            for bi, t in enumerate(idea_texts):
                for w in self._windows_of(t):
                    all_chunks.append(w); owners.append(bi)
            chunk_logits = self._score_logits(all_chunks)
            B = len(idea_texts)
            rows = [[] for _ in range(B)]
            for logit, bi in zip(chunk_logits.tolist(), owners): rows[bi].append(logit)
            # aggregate
            import math as _m
            maxW = max(len(r) for r in rows)
            arr = torch.full((B, maxW), float("nan"))
            for i, row in enumerate(rows): arr[i, :len(row)] = torch.tensor(row)
            mask = ~torch.isnan(arr); arr2 = torch.where(mask, arr, torch.zeros_like(arr))
            meanv = (arr2.sum(dim=1) / mask.sum(dim=1).clamp(min=1))
            if self.agg == "median":
                out = []
                for i in range(B):
                    xs = arr[i][mask[i]].tolist()
                    out.append(float(np.median(xs) if xs else 0.0))
                raw = torch.tensor(out)
            elif self.agg == "max":
                raw = torch.where(mask, arr2, torch.full_like(arr2, -1e9)).max(dim=1).values
            else:
                raw = meanv
        norm01 = torch.clamp((raw - 1.0)/9.0, 0.0, 1.0)
        prob = None
        if self.calib_fn is not None:
            p = [self.calib_fn(float(x)) for x in norm01.tolist()]
            prob = torch.tensor(p, dtype=torch.float32, device=norm01.device)
        return raw.float(), norm01.float(), (prob.float() if prob is not None else None)

# ================================================================
# RAG / prompts
# ================================================================
@dataclass
class SeedItem:
    topic: str
    problem: str
    sources: List[str]
    constraints: str = ""
    context: List[Dict[str, Any]] = None
    input_concepts: List[str] = None

def load_seeds(path: str, limit: Optional[int] = None) -> List[SeedItem]:
    seeds: List[SeedItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            obj = json.loads(ln)
            seeds.append(SeedItem(
                topic=obj.get("topic",""),
                problem=obj.get("problem",""),
                sources=obj.get("sources",[]) or [],
                constraints=obj.get("constraints",""),
                context=obj.get("context",[]) or [],
                input_concepts=obj.get("input_concepts",[]) or None,
            ))
            if limit is not None and len(seeds) >= limit: break
    if not seeds: raise RuntimeError(f"no seeds at {path}")
    return seeds

RAG_CONTEXT_TEMPLATE = "\n---\nRAG_CONTEXT:\n{ctx}\n(Use the context only as background evidence. Do not copy; synthesize novel ideas.)\n"
def build_ctx_block(seed: Dict[str, Any], max_docs: int = 5) -> str:
    ctx_lines = []
    for d in seed.get("context", [])[:max_docs]:
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()
        if not title and not abstract: continue
        ctx_lines.append(f"- {title}\n  {abstract}")
    return RAG_CONTEXT_TEMPLATE.format(ctx="\n".join(ctx_lines)) if ctx_lines else ""

SCHEMA_JSON = json.dumps({
    "input_concepts": ["..."],
    "new_concepts": ["..."],
    "bridge_rationale": "...",
    "plan": "...",
    "risks": ["..."],
    "title": "...",
    "abstract": "..."
}, ensure_ascii=False)

OBSERVER_JSON_PROMPT = (
    "You are a creative research ideator.\n"
    "Given the TOPIC, PROBLEM, and SOURCE DOMAINS, propose ONE research idea.\n"
    "Respond ONLY in valid JSON matching the schema.\n\n"
    "TOPIC: {topic}\n"
    "PROBLEM: {problem}\n"
    "SOURCES: {sources}\n"
    "CONSTRAINTS: {constraints}\n\n"
    "{input_hint}"
    "Schema (keys required): {schema}\n\n"
    "Requirements: encourage bisociation (combine distant concepts).\n"
    "Return JSON only. No prose outside JSON.\n\n"
    "{rag_block}"
)
PIONEER_REFINE_JSON_PROMPT = (
    "You are a research editor improving feasibility and clarity.\n"
    "Refine the JSON idea to reduce risks and increase feasibility, while preserving novelty.\n"
    "Use the BRIDGE_HINTS to explicitly incorporate cross-domain connections.\n"
    "Return JSON only. No extra text.\n\n"
    "BRIDGE_HINTS: {bridge_hints}\n\n"
    "{idea_json}\n\n{rag_block}"
)

def _extract_fields(seed: SeedItem) -> List[str]:
    found = []
    for d in seed.context or []:
        for f in d.get("fields", []) or []:
            if f and f not in found: found.append(f)
    return found

def _guess_input_concepts(seed: SeedItem, observer_json: Optional[Dict[str, Any]]) -> List[str]:
    if observer_json and isinstance(observer_json.get("input_concepts"), list):
        return [str(x).strip() for x in observer_json.get("input_concepts") if str(x).strip()]
    if seed.input_concepts:
        return [str(x).strip() for x in seed.input_concepts if str(x).strip()]
    toks = re.split(r"[,/]|\\band\\b|\\+|\\|", seed.topic.lower())
    toks = [t.strip() for t in toks if t.strip()]
    return toks[:4]

def build_bridge_hints(seed: SeedItem, observer_json: Optional[Dict[str, Any]], max_hints: int = 3) -> List[str]:
    fields = _extract_fields(seed)
    base = _guess_input_concepts(seed, observer_json)
    pool = [f for f in fields if f and f.lower() not in {c.lower() for c in base}] or fields
    random.shuffle(pool)
    picks = pool[:max_hints]
    hints = [f"Leverage domain '{f}' to connect with input concepts {base}." for f in picks]
    fallback = [("control theory","text-to-policy distillation"),
                ("graph signal processing","causal discovery"),
                ("program synthesis","embodied evaluation"),
                ("neurosymbolic reasoning","active learning")]
    while len(hints) < max_hints and fallback:
        a,b = fallback.pop(random.randrange(len(fallback)))
        hints.append(f"Cross with {a} and {b} to induce bisociation.")
    return hints[:max_hints]

# ================================================================
# Generation + reward scaling
# ================================================================
@dataclass
class GenCfg:
    min_new_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float

class EMAStats:
    def __init__(self, decay: float = 0.99):
        self.decay=decay; self._mean=None; self._var=None; self._eps=1e-6
    def update(self, x: torch.Tensor):
        x = x.detach().float().mean()
        if self._mean is None:
            self._mean = x; self._var = torch.tensor(1.0, device=x.device)
        else:
            self._mean = self.decay*self._mean + (1-self.decay)*x
            self._var  = self.decay*self._var  + (1-self.decay)*(x-self._mean).pow(2)
    @property
    def mean(self): return float(self._mean.item() if self._mean is not None else 0.0)
    @property
    def std(self):  return float(torch.sqrt((self._var or torch.tensor(1.0))+self._eps).item())

def build_reward_scaler(mode: str = "raw_zscore_tanh", beta: float = 1.5):
    ema_raw = EMAStats(0.99); ema_prob = EMAStats(0.99)
    def scale(raw: torch.Tensor, n01: torch.Tensor, prob: Optional[torch.Tensor]) -> List[torch.Tensor]:
        m = (mode or "raw_zscore_tanh")
        b = float(beta)
        if m == "raw_identity":
            r = raw.float()
        elif m == "raw_zscore_tanh":
            ema_raw.update(raw); z = (raw.float()-ema_raw.mean)/max(ema_raw.std,1e-6); r = torch.tanh(z*b)
        elif m == "norm01_identity":
            r = n01.float()
        elif m == "norm01_zscore_tanh":
            ema_raw.update(n01); z = (n01.float()-ema_raw.mean)/max(ema_raw.std,1e-6); r = torch.tanh(z*b)
        elif m == "calibrated_prob" and prob is not None:
            ema_prob.update(prob); z = (prob.float()-ema_prob.mean)/max(ema_prob.std,1e-6); r = torch.tanh(z*b)
        else:
            r = n01.float()
        return [ri.detach() for ri in r]
    return scale

# ================================================================
# PPO trainer helpers
# ================================================================
def _record_to_prompt(rec: Dict[str, Any]) -> Optional[str]:
    if not isinstance(rec, dict): return None
    if "prompt" in rec and isinstance(rec["prompt"], str) and rec["prompt"].strip():
        return rec["prompt"].strip()
    topic = rec.get("topic"); problem = rec.get("problem"); constraints = rec.get("constraints"); sources = rec.get("sources"); ctx_list = rec.get("context")
    if isinstance(sources, list): src_txt = ", ".join(map(str, sources))
    elif isinstance(sources, str): src_txt = sources
    else: src_txt = ""
    ctx_lines = []
    if isinstance(ctx_list, list):
        for c in ctx_list[:5]:
            if isinstance(c, dict):
                t=c.get("title",""); a=c.get("abstract",""); fields=c.get("fields",[])
                fields_str=", ".join(fields) if isinstance(fields,list) else str(fields)
                ctx_lines.append(f"- {t} | fields: {fields_str} | abs: {a}")
            else:
                ctx_lines.append(f"- {str(c)}")
    headline = problem or topic or "Creative research idea generation"
    base = [f"Topic: {topic}" if topic else None,
            f"Problem: {problem}" if problem else None,
            f"Sources/Fields: {src_txt}" if src_txt else None,
            f"Constraints: {constraints}" if constraints else None]
    base = [b for b in base if b]
    ctx_block = "Context papers:\n" + "\n".join(ctx_lines[:10]) if ctx_lines else ""
    prompt = (f"{headline}\n" + ("\n".join(base)+"\n" if base else "") +
              (ctx_block+"\n\n" if ctx_block else "") +
              "Give me three concrete, novel, and feasible research ideas that bridge distant fields above. "
              "Each idea must include: (1) title, (2) 2–3 sentence rationale, (3) minimal viable experiment with metrics, (4) risks+mitigations.")
    return prompt

def _load_prompts_from_jsonl(path: str, max_seeds: Optional[int]) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                p = _record_to_prompt(rec)
                if p and p.strip():
                    prompts.append(p.strip())
            except json.JSONDecodeError:
                if line.strip():
                    prompts.append(line.strip())
    random.shuffle(prompts)
    if max_seeds and max_seeds > 0:
        prompts = prompts[:max_seeds]

    # ★ ここを強化：必ず十分な個数にする
    base_prompt = "Give me three creative research ideas that connect LLMs × Reinforcement Learning."
    if len(prompts) < 32:
        need = 32 - len(prompts)
        prompts.extend([base_prompt] * need)

    return prompts

def _build_hfdataset(prompts: List[str]):
    return HFDataset.from_dict({"prompt": prompts})

def _build_ppo_trainer(model, tokenizer, seeds_path: str, max_seeds: Optional[int],
                       learning_rate: float, batch_size: int, mini_batch_size: int, ppo_epochs: int,
                       lora_r: int, lora_alpha: int, lora_dropout: float):
    if not _HAS_TRL:
        print("[DEBUG] TRL not available.")
        return None

    prompts = _load_prompts_from_jsonl(seeds_path, max_seeds)
    dataset = _build_hfdataset(prompts)

    # Optional LoRA
    if lora_r and lora_r>0 and _HAS_PEFT:
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                              bias="none", task_type="CAUSAL_LM",
                              target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
        policy = model.pretrained_model
        policy = get_peft_model(policy, lora_cfg)
        model.pretrained_model = policy
        try: model.pretrained_model.config.use_cache = False
        except Exception: pass

    # PPO config
    ppo_config = PPOConfig(learning_rate=learning_rate, batch_size=batch_size, mini_batch_size=mini_batch_size,seed=42)
    if hasattr(ppo_config, "use_reference_model"): ppo_config.use_reference_model = False
    if hasattr(ppo_config, "ppo_epochs"): ppo_config.ppo_epochs = ppo_epochs
    elif hasattr(ppo_config, "epochs"): ppo_config.epochs = ppo_epochs

    # Try multiple constructor signatures (robust to TRL API drift)
    for attempt in range(4):
        try:
            if attempt == 0:
                return PPOTrainer(config=ppo_config, model=model, ref_model=None,
                                  tokenizer=tokenizer, dataset=dataset, data_collator=None)
            elif attempt == 1:
                return PPOTrainer(ppo_config, model, None, tokenizer, dataset, None)
            elif attempt == 2:
                return PPOTrainer(model=model, tokenizer=tokenizer, dataset=dataset, data_collator=None)
            else:
                # dataset=None path (rarely needed, but increases init pass rate)
                return PPOTrainer(config=ppo_config, model=model, ref_model=None,
                                  tokenizer=tokenizer, dataset=None, data_collator=None)
        except Exception as e:
            print(f"[DEBUG] PPOTrainer init attempt {attempt} failed: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
    return None

def generate_texts_with_trl(ppo_trainer, prompts: List[str], cfg: GenCfg) -> List[str]:
    tokenizer = getattr(ppo_trainer, "tokenizer", None)
    model = getattr(ppo_trainer, "model", None)
    if tokenizer is None or model is None:
        raise RuntimeError("trainer missing tokenizer/model")
    device = getattr(getattr(ppo_trainer, "accelerator", None), "device", None)
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    enc = {k:v.to(device) for k,v in enc.items()}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    gen = model.pretrained_model.generate(
        **enc, do_sample=True, top_p=cfg.top_p, top_k=cfg.top_k, temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty, min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens, pad_token_id=pad_id, eos_token_id=eos_id,
    )
    input_len = enc["input_ids"].shape[1]
    new_tokens = gen[:, input_len:]
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return texts

# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--trust-remote-code", action="store_true")

    # PPO
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--mini-batch-size", type=int, default=1)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=2000)
    ap.add_argument("--swap-every", type=int, default=5)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Generation
    ap.add_argument("--min-new-tokens", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)

    # IRM
    ap.add_argument("--irm-model-dir", type=str, default="./IRM/irm_sci_huber_z_splitstats")
    ap.add_argument("--irm-max-len", type=int, default=512)
    ap.add_argument("--irm-quantization", type=str, default="none", choices=["none","8bit","4bit"])
    ap.add_argument("--irm-torch-dtype", type=str, default="auto", choices=["auto","float16","bfloat16","float32"])
    ap.add_argument("--irm-use-sliding", action="store_true")
    ap.add_argument("--irm-stride-ratio", type=float, default=0.5)
    ap.add_argument("--irm-agg", type=str, default="median", choices=["mean","median","max"])
    ap.add_argument("--irm-calib-path", type=str, default=None)

    # Reward scaling
    ap.add_argument("--reward-scale-mode", type=str, default="calibrated_prob",
                    choices=["raw_identity","raw_zscore_tanh","norm01_identity","norm01_zscore_tanh","calibrated_prob"])
    ap.add_argument("--reward-scale-beta", type=float, default=1.5)

    # Data
    ap.add_argument("--seeds-path", type=str, default="./data/research_seeds.jsonl")
    ap.add_argument("--max-seeds", type=int, default=2000)

    # Logging / run meta
    ap.add_argument("--group", type=str, default="cory-crea-bridge")
    ap.add_argument("--run-name", type=str, default=f"cory-crea-bridge-{int(time.time())}")
    ap.add_argument("--disable-wandb", action="store_true")

    # Deepspeed via Accelerate
    ap.add_argument("--use-deepspeed", action="store_true",
                    help="Set ACCELERATE_USE_DEEPSPEED=1 for PPO/Accelerate backend (requires accelerate config).")

    args = ap.parse_args()

    # Deepspeed (via accelerate)
    if args.use_deepspeed:
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
        print("[INFO] Using Deepspeed via Accelerate. Ensure your accelerate config/yaml enables Deepspeed.")

    # wandb
    if not args.disable_wandb and _HAS_WANDB:
        try:
            wandb.init(project=args.group, name=args.run_name)
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")

    # Tokenizer (left padding)
    gen_tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    gen_tokenizer.padding_side = "left"
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    # IRM
    irm_reward = IRMReward(
        irm_model_dir=args.irm_model_dir, max_length=args.irm_max_len,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        quantization=args.irm_quantization, torch_dtype=args.irm_torch_dtype,
        use_sliding=args.irm_use_sliding, stride_ratio=args.irm_stride_ratio,
        agg=args.irm_agg, calib_path=args.irm_calib_path,
    )
    scale_rewards = build_reward_scaler(args.reward_scale_mode, args.reward_scale_beta)

    # Seeds
    try:
        seeds = load_seeds(args.seeds_path, limit=args.max_seeds)
    except Exception as e:
        print(f"[WARN] load_seeds failed: {e}")
        seeds = []

    # PPO model/trainer
    ppo_trainer = None
    if _HAS_TRL:
        policy = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.model_name, trust_remote_code=args.trust_remote_code,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else
                         (torch.float16 if torch.cuda.is_available() else torch.float32)),
            device_map="auto",
        )
        try:
            policy.pretrained_model.config.use_cache = False
        except Exception:
            pass

        ppo_trainer = _build_ppo_trainer(
            policy, gen_tokenizer,
            seeds_path=args.seeds_path, max_seeds=args.max_seeds,
            learning_rate=args.learning_rate, batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size, ppo_epochs=args.ppo_epochs,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout
        )

    # Generation configs
    cfg_obs = GenCfg(args.min_new_tokens, args.max_new_tokens, args.temperature, args.top_p, args.top_k, 1.0)
    cfg_pio = GenCfg(128, min(700, args.max_new_tokens), 0.7, 0.9, args.top_k, 1.05)

    # Prompt builders
    def build_observer_prompt(s: SeedItem, ctx_docs: int = 5) -> str:
        rag_block = build_ctx_block(s.__dict__, max_docs=ctx_docs)
        ics = (s.input_concepts or [])
        input_hint = f"INPUT_CONCEPTS_HINT: {', '.join(ics)}\n" if ics else ""
        return OBSERVER_JSON_PROMPT.format(
            topic=s.topic, problem=s.problem, sources=", ".join(s.sources),
            constraints=s.constraints or "", input_hint=input_hint,
            rag_block=rag_block, schema=SCHEMA_JSON
        )

    def batch_iterator(data, batch_size):
        i=0; pool=data[:]; random.shuffle(pool)
        while True:
            if not pool: raise RuntimeError("no seeds")
            if batch_size>len(pool): yield random.choices(pool, k=batch_size)
            else:
                if i+batch_size>len(pool): random.shuffle(pool); i=0
                yield pool[i:i+batch_size]; i+=batch_size

    # === NO-PPO fallback ===
    if ppo_trainer is None:
        print("[INFO] PPO unavailable -> NO-PPO (generation + IRM scoring only).")
        head = seeds[:max(1, min(8, len(seeds) or 1))]
        obs_prompts = [build_observer_prompt(s, 5) for s in head]

        gen_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, trust_remote_code=args.trust_remote_code,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else
                         (torch.float16 if torch.cuda.is_available() else torch.float32)),
            device_map="auto"
        )

        def _simple_generate(prompts: List[str], cfg: GenCfg):
            dev = next(gen_model.parameters()).device
            enc = gen_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            enc = {k:v.to(dev) for k,v in enc.items()}
            pad_id = gen_tokenizer.pad_token_id or gen_tokenizer.eos_token_id
            eos_id = gen_tokenizer.eos_token_id
            out = gen_model.generate(**enc, do_sample=True, top_p=cfg.top_p, top_k=cfg.top_k,
                                     temperature=cfg.temperature, repetition_penalty=cfg.repetition_penalty,
                                     min_new_tokens=cfg.min_new_tokens, max_new_tokens=cfg.max_new_tokens,
                                     pad_token_id=pad_id, eos_token_id=eos_id)
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            return gen_tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        resp_obs = _simple_generate(obs_prompts, cfg_obs)
        obs_jsons = [try_parse_json(t) for t in resp_obs]
        obs_valid = [x if (x and validate_schema(x)) else None for x in obs_jsons]

        hints = [build_bridge_hints(s, v, 3) for s, v in zip(head, obs_valid)]
        pio_prompts=[]
        for s, idea, hs in zip(head, obs_valid, hints):
            rag_block = build_ctx_block(s.__dict__, 5)
            pio_prompts.append(PIONEER_REFINE_JSON_PROMPT.format(
                bridge_hints="; ".join(hs) if hs else "", idea_json=json.dumps(idea or {}, ensure_ascii=False),
                rag_block=rag_block
            ))
        resp_pio = _simple_generate(pio_prompts, cfg_pio)

        def _mktexts(seeds, parseds, raws):
            xs=[]
            for s, p, r in zip(seeds, parseds, raws):
                t = make_irm_text_from_idea_relaxed(p) if p else fallback_irm_text_from_raw_response(r, s.problem)
                xs.append(t)
            return xs

        obs_texts = _mktexts(head, obs_jsons, resp_obs)
        pio_texts = _mktexts(head, [try_parse_json(t) for t in resp_pio], resp_pio)
        r1, n1, p1 = irm_reward.score(obs_texts)
        r2, n2, p2 = irm_reward.score(pio_texts)
        allr = torch.cat([r1, r2], 0)
        print(f"[NO-PPO] IRM raw mean={allr.mean().item():.4f}, std={allr.std().item():.4f}")

        os.makedirs("./outputs/crea_bridge_last", exist_ok=True)
        with open("./outputs/crea_bridge_last/samples_no_ppo.jsonl","w",encoding="utf-8") as f:
            for o,p in zip(resp_obs, resp_pio):
                f.write(json.dumps({"observer": o, "pioneer": p}, ensure_ascii=False)+"\n")
        print("Saved: ./outputs/crea_bridge_last/samples_no_ppo.jsonl")
        return

    # === PPO mode ===
    print(f"[DEBUG] _HAS_TRL={_HAS_TRL}, _HAS_PEFT={_HAS_PEFT}, _HAS_DS={_HAS_DS}")
    print(f"[DEBUG] seeds_path exists: {os.path.isfile(args.seeds_path)}")

    reward_hist=[]
    def moving_avg(xs, k=20):
        out=[]; s=0.0; q=[]
        for x in xs:
            q.append(x); s+=x
            if len(q)>k: s-=q.pop(0)
            out.append(s/len(q))
        return out

    get_batch = batch_iterator(seeds, args.batch_size)
    for step in tqdm(range(args.total_steps), desc="train"):
        batch = next(get_batch)

        # Observer
        obs_prompts = [build_observer_prompt(s, 5) for s in batch]
        resp_obs = generate_texts_with_trl(ppo_trainer, obs_prompts, cfg_obs)
        obs_jsons = [try_parse_json(t) for t in resp_obs]
        obs_valid = [x if (x and validate_schema(x)) else None for x in obs_jsons]

        # Bridger -> Pioneer
        bridge_hints_batch = [build_bridge_hints(s, idea, 3) for s, idea in zip(batch, obs_valid)]
        pio_prompts=[]
        for s, idea, hints in zip(batch, obs_valid, bridge_hints_batch):
            rag_block = build_ctx_block(s.__dict__, 5)
            pio_prompts.append(PIONEER_REFINE_JSON_PROMPT.format(
                bridge_hints="; ".join(hints) if hints else "",
                idea_json=json.dumps(idea or {}, ensure_ascii=False),
                rag_block=rag_block
            ))
        resp_pio = generate_texts_with_trl(ppo_trainer, pio_prompts, cfg_pio)

        # IRM scores
        def _mktexts(seeds, parseds, raws):
            xs=[]
            for s, p, r in zip(seeds, parseds, raws):
                t = make_irm_text_from_idea_relaxed(p) if p else fallback_irm_text_from_raw_response(r, s.problem)
                xs.append(t)
            return xs
        obs_texts = _mktexts(batch, obs_jsons, resp_obs)
        pio_texts = _mktexts(batch, [try_parse_json(t) for t in resp_pio], resp_pio)
        raw_obs_t, n01_obs_t, prob_obs_t = irm_reward.score(obs_texts)
        raw_pio_t, n01_pio_t, prob_pio_t = irm_reward.score(pio_texts)

        # PPO steps (observer + periodically pioneer)
        def _ppo_step(prompts, responses):
            for attempt in range(2):
                try:
                    return ppo_trainer.step(prompts=prompts, responses=responses)
                except TypeError:
                    try:
                        return ppo_trainer.step(prompts, responses)
                    except Exception:
                        if attempt==1: return {}
                except Exception:
                    return {}
            return {}

        _ = _ppo_step(obs_prompts, resp_obs)
        if (step % max(1, args.swap_every)) == 0:
            _ = _ppo_step(pio_prompts, resp_pio)

        # logging
        raw_all = torch.cat([raw_obs_t, raw_pio_t], 0)
        n01_all = torch.cat([n01_obs_t, n01_pio_t], 0)
        prob_all = (torch.cat([prob_obs_t, prob_pio_t], 0) if (prob_obs_t is not None and prob_pio_t is not None) else None)
        scaled_all = torch.stack(build_reward_scaler(args.reward_scale_mode, args.reward_scale_beta)(raw_all, n01_all, prob_all))
        reward_hist.append(float(scaled_all.mean().item()))
        smooth = moving_avg(reward_hist, 20)[-1]

        if _HAS_WANDB and (not args.disable_wandb):
            try:
                wandb.log({
                    "env/reward_raw_mean": float(raw_all.mean().item()),
                    "env/reward_raw_std": float(raw_all.std().item()),
                    "env/reward_mean_scaled": float(scaled_all.mean().item()),
                    "env/reward_std_scaled": float(scaled_all.std().item()),
                    "env/reward_smooth_scaled": smooth,
                    "trainer/step": step,
                    "env/irm_sliding": float(1.0 if args.irm_use_sliding else 0.0),
                    "env/irm_stride_ratio": float(args.irm_stride_ratio),
                })
            except Exception as e:
                print(f"[WARN] wandb.log failed: {e}")

    print(">> Training finished.")
    out_dir = "./outputs/crea_bridge_last"
    os.makedirs(out_dir, exist_ok=True)
    try:
        ppo_trainer.save_pretrained(out_dir)
        gen_tokenizer.save_pretrained(out_dir)
    except Exception:
        pass
    print(f"Saved model to: {out_dir}")

if __name__ == "__main__":
    main()

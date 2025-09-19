#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CREA-Bridge: CORY + Bridger + IRM + PPO (single-file runner)
- 既存の cory_withIRM_rag.py【構造・関数名は極力互換】をベースに、Bridger エージェントを追加
- Bridger は Observer の初期案と RAG 文脈を読み、遠距離概念（bisociation）を提案し、
  Pioneer の入力 JSON に追加ヒント（BRIDGE_HINTS）として注入する
- IRM で Observer/Pioneer の両方をスコアリングし、PPO で更新

実行例:
python cory_crea_bridge.py \
  --model_name gpt2 \
  --irm_model_dir ./IRM/irm_iclr_model \
  --seeds_path ./data/research_seeds.jsonl \
  --total_steps 1000

依存:
  - trainer_builder.py （同ディレクトリに置く）【17†source】
  - 研究シード research_seeds.jsonl （search_to_seeds.py で作成）【16†source】
  - IRM 学習済み分類器（AutoModelForSequenceClassification, 1..10 予測）【15†source】
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# tqdm の安全 import（notebook/CLI 双方で落ちない）
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it=None, **kw):
        return it if it is not None else range(kw.get("total", 0))

import torch
import tyro
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import is_bitsandbytes_available
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

# ================================================================
# 既存 cory_withIRM_rag.py のユーティリティを踏襲（必要最小限を同梱）
# 参照: REQUIRED_KEYS / JSON整形 / IRMReward / EMAStats 等は互換
# ================================================================
REQUIRED_KEYS = [
    "input_concepts", "new_concepts", "bridge_rationale",
    "plan", "risks", "title", "abstract"
]

_JSON_TITLE_RE = re.compile(r'"title"\s*:\s*"([^"]+)"', re.S)
_JSON_ABS_RE   = re.compile(r'"abstract"\s*:\s*"([^"]+)"', re.S)


def _strip_nonjson_tail(s: str) -> str:
    if not s:
        return s
    i = s.find('{'); j = s.rfind('}')
    if i != -1 and j != -1 and j > i:
        return s[i:j+1]
    return s


def try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s1 = _strip_nonjson_tail(s).replace("\r", "")
    s1 = re.sub(r',\s*([}\]])', r'\1', s1)
    try:
        return json.loads(s1)
    except Exception:
        return None


def validate_schema(obj: Dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    for k in REQUIRED_KEYS:
        if k not in obj:
            return False
        v = obj[k]
        if isinstance(v, str) and len(v.strip()) < 5:
            return False
        if isinstance(v, list) and len(v) == 0:
            return False
    ic = obj.get("input_concepts", []); nc = obj.get("new_concepts", [])
    if isinstance(ic, list) and isinstance(nc, list):
        if set(map(str.lower, ic)) == set(map(str.lower, nc)):
            return False
    return True


def make_irm_text_from_idea_relaxed(idea: Dict[str, Any]) -> str:
    if not isinstance(idea, dict):
        return ""
    t = str(idea.get("title", "") or "").strip()
    a = str(idea.get("abstract", "") or "").strip()
    b = str(idea.get("bridge_rationale", "") or "").strip()
    p = str(idea.get("plan", "") or "").strip()
    if not (t or a or b or p):
        return ""
    parts = []
    if t: parts.append(t)
    if a: parts.append(a)
    if b: parts.append(f"Bridge: {b}")
    if p: parts.append(f"Plan: {p}")
    return "\n\n".join(parts)


def fallback_irm_text_from_raw_response(resp_text: str, seed_problem: str = "") -> str:
    if not resp_text:
        head = seed_problem.strip() or "Generated Idea"
        return f"{head}\n\nEmpty response."
    s = _strip_nonjson_tail(resp_text)
    mt = _JSON_TITLE_RE.search(s)
    ma = _JSON_ABS_RE.search(s)
    if mt or ma:
        t = mt.group(1).strip() if mt else ""
        a = ma.group(1).strip() if ma else s[:1000]
        return (t + "\n\n" + a).strip()
    body = s.strip()[:1600]
    head = seed_problem.strip() or "Generated Idea"
    return f"{head}\n\n{body}"


class IRMReward:
    def __init__(
        self,
        irm_model_dir: str,
        max_length: int = 512,
        device: torch.device | None = None,
        quantization: str = "none",
        torch_dtype: str | None = "auto",
        device_map: str = "auto",
        low_cpu_mem_usage: bool = True,
        offload_folder: str | None = None,
        max_memory: dict | None = None,
    ):
        self.max_length = max_length
        self._uses_device_map = False
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tok = AutoTokenizer.from_pretrained(irm_model_dir, use_fast=True)

        if torch_dtype == "float16":
            _dtype = torch.float16
        elif torch_dtype == "bfloat16":
            _dtype = torch.bfloat16
        elif torch_dtype == "float32":
            _dtype = torch.float32
        else:
            _dtype = None

        def _maybe_dtype(kw: dict):
            if _dtype is not None and "torch_dtype" not in kw:
                kw["torch_dtype"] = _dtype
            return kw

        if quantization in ("8bit", "4bit"):
            if not is_bitsandbytes_available() or BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes が必要です。`pip install -U bitsandbytes` を実行してください。")
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=(quantization == "8bit"),
                load_in_4bit=(quantization == "4bit"),
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            kw = {
                "quantization_config": bnb_cfg,
                "device_map": device_map or "auto",
                "low_cpu_mem_usage": low_cpu_mem_usage,
            }
            if max_memory is not None:
                kw["max_memory"] = max_memory
            if offload_folder is not None:
                kw["offload_folder"] = offload_folder
            _maybe_dtype(kw)
            self.model = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, **kw)
            self._uses_device_map = True
            self.model.eval()
            return

        kw = {"low_cpu_mem_usage": low_cpu_mem_usage}
        if max_memory is not None:
            kw["max_memory"] = max_memory
        _maybe_dtype(kw)
        try:
            m = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, **kw)
            m.to(self.device).eval()
            self.model = m
        except RuntimeError:
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.device = torch.device("cpu")
            m = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, **kw)
            m.to(self.device).eval()
            self.model = m

    @torch.no_grad()
    def score(self, idea_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(
            idea_texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        if not self._uses_device_map:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits.squeeze(-1).detach().float()
        raw = logits
        norm01 = torch.clamp((raw - 1.0) / 9.0, 0.0, 1.0)
        return raw, norm01


class EMAStats:
    def __init__(self, decay: float = 0.99):
        self.decay = decay; self._mean = None; self._var = None; self._eps = 1e-6
    def update(self, x: torch.Tensor):
        x = x.detach().float().mean()
        if self._mean is None:
            self._mean = x; self._var = torch.tensor(1.0, device=x.device)
        else:
            self._mean = self.decay * self._mean + (1 - self.decay) * x
            self._var = self.decay * self._var + (1 - self.decay) * (x - self._mean).pow(2)
    @property
    def mean(self): return float(self._mean.item() if self._mean is not None else 0.0)
    @property
    def std(self):  return float(torch.sqrt((self._var or torch.tensor(1.0)) + self._eps).item())


# ================================================================
# RAG + プロンプト（既存の設計を踏襲）
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
                input_concepts=obj.get("input_concepts", []) or None,
            ))
            if limit is not None and len(seeds) >= limit:
                break
    assert len(seeds) > 0, f"no seeds at {path}"
    return seeds


RAG_CONTEXT_TEMPLATE = """
---
RAG_CONTEXT:
{ctx}
(Use the context only as background evidence. Do not copy; synthesize novel ideas.)
"""


def build_ctx_block(seed: Dict[str, Any], max_docs: int = 5) -> str:
    ctx_lines = []
    for d in seed.get("context", [])[:max_docs]:
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()
        if not title and not abstract:
            continue
        ctx_lines.append(f"- {title}\n  {abstract}")
    return RAG_CONTEXT_TEMPLATE.format(ctx="\n".join(ctx_lines)) if ctx_lines else ""


OBSERVER_JSON_PROMPT = (
    "You are a creative research ideator.\n"
    "Given the TOPIC, PROBLEM, and SOURCE DOMAINS, propose ONE research idea.\n"
    "Respond ONLY in valid JSON matching the schema.\n\n"
    "TOPIC: {topic}\n"
    "PROBLEM: {problem}\n"
    "SOURCES: {sources}\n"
    "CONSTRAINTS: {constraints}\n\n"
    "{input_hint}"
    "Schema (keys required): {\"input_concepts\":[\"...\"], \"new_concepts\":[\"...\"], "
    "\"bridge_rationale\":\"...\", \"plan\":\"...\", \"risks\":[\"...\"], "
    "\"title\":\"...\", \"abstract\":\"...\"}\n\n"
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


def build_observer_prompt(seed: SeedItem, ctx_docs: int = 5) -> str:
    rag_block = build_ctx_block(seed.__dict__, max_docs=ctx_docs)
    ics = (seed.input_concepts or [])
    input_hint = f"INPUT_CONCEPTS_HINT: {', '.join(ics)}\n" if ics else ""
    return OBSERVER_JSON_PROMPT.format(
        topic=seed.topic,
        problem=seed.problem,
        sources=", ".join(seed.sources),
        constraints=seed.constraints or "",
        input_hint=input_hint,
        rag_block=rag_block,
    )


# ================================================================
# CREA-Bridge: Bridger エージェント
# ================================================================
@dataclass
class BridgerCfg:
    max_hints: int = 3
    prefer_new_fields: bool = True
    allow_fallback_catalog: bool = True


_BRIDGE_FALLBACK = [
    # 交差領域カタログ（安全なデフォルト）
    ("control theory", "text-to-policy distillation"),
    ("graph signal processing", "causal discovery"),
    ("program synthesis", "embodied evaluation"),
    ("neurosymbolic reasoning", "toolformer planning"),
    ("mechanistic interpretability", "active learning"),
    ("edge robotics", "federated RL"),
]


def _extract_fields(seed: SeedItem) -> List[str]:
    found = []
    for d in seed.context or []:
        for f in d.get("fields", []) or []:
            if f and f not in found:
                found.append(f)
    return found


def _guess_input_concepts(seed: SeedItem, observer_json: Optional[Dict[str, Any]]) -> List[str]:
    if observer_json and isinstance(observer_json.get("input_concepts"), list):
        return [str(x).strip() for x in observer_json.get("input_concepts") if str(x).strip()]
    if seed.input_concepts:
        return [str(x).strip() for x in seed.input_concepts if str(x).strip()]
    # topic から緩く抽出（カンマ/スラッシュ分割）
    toks = re.split(r"[,/]|\band\b|\+|\|", seed.topic.lower())
    toks = [t.strip() for t in toks if t.strip()]
    return toks[:4]


def _pick_distant_fields(all_fields: List[str], used_concepts: List[str], k: int) -> List[str]:
    # シンプル: seed.context の分野から未使用のものを優先、無ければランダム
    pool = [f for f in all_fields if f and f.lower() not in {c.lower() for c in used_concepts}]
    if not pool:
        return random.sample(all_fields, min(k, len(all_fields))) if all_fields else []
    random.shuffle(pool)
    return pool[:k]


def build_bridge_hints(seed: SeedItem, observer_json: Optional[Dict[str, Any]], cfg: BridgerCfg) -> List[str]:
    fields = _extract_fields(seed)
    base_concepts = _guess_input_concepts(seed, observer_json)

    # 1) context にある別分野からピック
    picks = _pick_distant_fields(fields, base_concepts, cfg.max_hints)
    hints: List[str] = []
    for f in picks:
        hints.append(f"Leverage domain '{f}' to connect with input concepts {base_concepts}.")

    # 2) 予備カタログ
    if cfg.allow_fallback_catalog and len(hints) < cfg.max_hints:
        remain = cfg.max_hints - len(hints)
        cand = random.sample(_BRIDGE_FALLBACK, min(remain, len(_BRIDGE_FALLBACK)))
        for a, b in cand:
            hints.append(f"Cross with {a} and {b} to induce bisociation.")

    return hints[: cfg.max_hints]


# ================================================================
# 生成 / 報酬スケール / 設定
# ================================================================
@dataclass
class GenCfg:
    min_new_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


@dataclass
class TrainArgs:
    model_name: str = "gpt2"
    trust_remote_code: bool = False

    # PPO
    learning_rate: float = 1e-5
    batch_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    kl_target: float = 0.15
    max_grad_norm: float = 1.0
    ratio_threshold: float = 0.2

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # generation
    min_new_tokens: int = 64
    max_new_tokens: int = 192
    top_k: int = 50
    top_p: float = 0.9
    temperature: float = 0.7
    repetition_penalty: float = 1.05

    # steps
    total_steps: int = 1000
    swap_every: int = 5

    # logs
    group: Optional[str] = "cory-crea-bridge"
    run_name: Optional[str] = f"cory-crea-bridge-{int(time.time())}"

    # IRM
    irm_model_dir: str = "./IRM/irm_iclr_model"
    irm_max_len: int = 512
    reward_scale_mode: str = "raw_zscore_tanh"  # 報酬一定化の緩和
    reward_scale_beta: float = 1.5

    # seeds
    seeds_path: str = "./data/research_seeds.jsonl"
    max_seeds: int = 100000

    # IRM quantization
    irm_quantization: str = "8bit"     # "none" | "8bit" | "4bit"
    irm_torch_dtype: str = "auto"      # "auto" | "float16" | "bfloat16" | "float32"
    irm_device_map: str = "auto"
    irm_low_cpu_mem_usage: bool = True
    irm_offload_folder: str | None = None
    irm_max_memory: dict | None = None

    # Bridger
    bridge_max_hints: int = 3


def build_reward_scaler(args: TrainArgs):
    ema_raw = EMAStats(decay=0.99)

    def scale_rewards(raw: torch.Tensor, norm01: torch.Tensor) -> List[torch.Tensor]:
        mode = args.reward_scale_mode
        beta = float(args.reward_scale_beta)
        if mode == "raw_identity":
            r = raw.float()
        elif mode == "raw_zscore_tanh":
            ema_raw.update(raw)
            mu, std = ema_raw.mean, max(ema_raw.std, 1e-6)
            z = (raw.float() - mu) / std
            r = torch.tanh(z * beta)
        elif mode == "norm01_identity":
            r = norm01.float()
        elif mode == "norm01_zscore_tanh":
            ema_raw.update(norm01)
            mu, std = ema_raw.mean, max(ema_raw.std, 1e-6)
            z = (norm01.float() - mu) / std
            r = torch.tanh(z * beta)
        else:
            r = norm01.float()
        # List[Tensor]（サンプル毎）で返して PPOTrainer にそのまま渡す
        return [ri.detach() for ri in r]

    return scale_rewards


# ================================================================
# 生成ヘルパ
# ================================================================

def generate_texts(ppo_trainer, prompts: List[str], cfg: GenCfg) -> List[str]:
    out = ppo_trainer.generate(
        prompts,
        do_sample=True,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        min_new_tokens=cfg.min_new_tokens,
        max_new_tokens=cfg.max_new_tokens,
        pad_token_id=ppo_trainer.tokenizer.eos_token_id,
        eos_token_id=ppo_trainer.tokenizer.eos_token_id,
    )
    if isinstance(out, list):
        return out
    try:
        seqs = out.sequences
        return ppo_trainer.tokenizer.batch_decode(seqs, skip_special_tokens=True)
    except Exception:
        if hasattr(out, 'sequences'):
            return [ppo_trainer.tokenizer.decode(x, skip_special_tokens=True) for x in out.sequences]
        return [str(out)]


# ================================================================
# メインループ
# ================================================================

def main(args: TrainArgs):
    # ログ
    wandb.init(project=args.group or "cory-crea-bridge", name=args.run_name)

    # Tokenizer（生成用）
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PPO トレーナ
    from trainer_builder import build_ppo_trainer  # 【17†source】
    ppo_trainer = build_ppo_trainer(
        args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        kl_target=args.kl_target,
        max_grad_norm=args.max_grad_norm,
        ratio_threshold=args.ratio_threshold,
        trust_remote_code=args.trust_remote_code,
        lora={
            "r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
            "bias": "none", "task_type": "CAUSAL_LM",
            # target_modules は trainer_builder 側で自動推定されます【17†source】
        },
        gen_defaults={
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
        },
    )

    device = ppo_trainer.accelerator.device

    # IRM Reward 構築（8bit/4bit による省メモリも可）
    irm_reward = IRMReward(
        irm_model_dir=args.irm_model_dir,
        max_length=args.irm_max_len,
        device=device,
        quantization=args.irm_quantization,
        torch_dtype=args.irm_torch_dtype,
        device_map=args.irm_device_map,
        low_cpu_mem_usage=args.irm_low_cpu_mem_usage,
        offload_folder=args.irm_offload_folder,
        max_memory=args.irm_max_memory,
    )

    scale_rewards = build_reward_scaler(args)
    seeds = load_seeds(args.seeds_path, limit=args.max_seeds)  # 【16†source】で作成したJSONL形式に対応

    # 生成設定
    cfg_obs = GenCfg(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.0,
    )
    cfg_pio = GenCfg(
        min_new_tokens=128,
        max_new_tokens=min(700, args.max_new_tokens),
        temperature=0.7,
        top_p=0.9,
        top_k=args.top_k,
        repetition_penalty=1.05,
    )

    reward_hist: List[float] = []

    # 重複を避けるロバストなバッチイテレータ（cory_withIRM_rag.py と同様の設計）
    def batch_iterator(data, batch_size):
        i = 0
        pool = data[:]
        random.shuffle(pool)
        while True:
            if len(pool) == 0:
                raise RuntimeError("no seeds")
            if batch_size > len(pool):
                yield random.choices(pool, k=batch_size)
            else:
                if i + batch_size > len(pool):
                    random.shuffle(pool)
                    i = 0
                yield pool[i:i+batch_size]
                i += batch_size

    get_batch = batch_iterator(seeds, args.batch_size)

    for step in tqdm(range(args.total_steps), desc="train"):
        batch = next(get_batch)

        # --- Observer ---
        obs_prompts = [build_observer_prompt(s, ctx_docs=5) for s in batch]
        resp_obs = generate_texts(ppo_trainer, obs_prompts, cfg_obs)
        obs_jsons = [try_parse_json(t) for t in resp_obs]
        obs_valid = [x if (x and validate_schema(x)) else None for x in obs_jsons]

        # --- Bridger ---
        bridge_hints_batch: List[List[str]] = []
        for s, idea in zip(batch, obs_valid):
            hints = build_bridge_hints(s, idea, BridgerCfg(max_hints=args.bridge_max_hints))
            bridge_hints_batch.append(hints)

        # --- Pioneer (BRIDGE_HINTS を注入) ---
        pio_prompts = []
        for s, idea, hints in zip(batch, obs_valid, bridge_hints_batch):
            rag_block = build_ctx_block(s.__dict__, max_docs=5)
            if idea is None:
                idea_json = json.dumps({}, ensure_ascii=False)
            else:
                idea_json = json.dumps(idea, ensure_ascii=False)
            pio_prompts.append(
                PIONEER_REFINE_JSON_PROMPT.format(
                    bridge_hints="; ".join(hints) if hints else "",
                    idea_json=idea_json,
                    rag_block=rag_block,
                )
            )

        resp_pio = generate_texts(ppo_trainer, pio_prompts, cfg_pio)
        pio_jsons = [try_parse_json(t) for t in resp_pio]
        pio_valid = [x if (x and validate_schema(x)) else None for x in pio_jsons]

        # --- IRM scores（空文字禁止：緩い整形＋フォールバック） ---
        idea_obs_texts: List[str] = []
        for s, parsed, raw in zip(batch, obs_jsons, resp_obs):
            txt = make_irm_text_from_idea_relaxed(parsed) if parsed else ""
            if not txt.strip():
                txt = fallback_irm_text_from_raw_response(raw, seed_problem=s.problem)
            idea_obs_texts.append(txt)

        idea_pio_texts: List[str] = []
        for s, parsed, raw in zip(batch, pio_jsons, resp_pio):
            txt = make_irm_text_from_idea_relaxed(parsed) if parsed else ""
            if not txt.strip():
                txt = fallback_irm_text_from_raw_response(raw, seed_problem=s.problem)
            idea_pio_texts.append(txt)

        # W&B デバッグ: 入力断片を可視化
        try:
            def _preview(xs: List[str], n=4):
                out = []
                for x in xs[:n]:
                    h = (x[:140] + '…') if len(x) > 140 else x
                    out.append((len(x), h))
                return out
            wandb.log({
                "debug/idea_obs_samples": wandb.Table(columns=["len","head"], data=_preview(idea_obs_texts, 4)),
                "debug/idea_pio_samples": wandb.Table(columns=["len","head"], data=_preview(idea_pio_texts, 4)),
                "debug/bridge_hints_sample": wandb.Table(columns=["hints"], data=[[" | ".join(bridge_hints_batch[0])]] if bridge_hints_batch else []),
            })
        except Exception:
            pass

        raw_obs_t, n01_obs_t = irm_reward.score(idea_obs_texts)
        raw_pio_t, n01_pio_t = irm_reward.score(idea_pio_texts)

        # --- PPO updates ---
        stats_obs = {}
        stats_pio = {}
        try:
            stats_obs = ppo_trainer.step(prompts=obs_prompts, responses=resp_obs,
                                         rewards=scale_rewards(raw_obs_t, n01_obs_t))
        except Exception:
            pass

        if (step % max(1, args.swap_every)) == 0:
            try:
                stats_pio = ppo_trainer.step(prompts=pio_prompts, responses=resp_pio,
                                             rewards=scale_rewards(raw_pio_t, n01_pio_t))
            except Exception:
                pass

        # --- logs ---
        raw_all = torch.cat([raw_obs_t, raw_pio_t], dim=0)
        n01_all = torch.cat([n01_obs_t, n01_pio_t], dim=0)
        scaled_all = torch.stack(scale_rewards(raw_all, n01_all))
        reward_hist.append(float(scaled_all.mean().item()))
        def moving_avg(xs: List[float], k: int = 20):
            out, s, q = [], 0.0, []
            for x in xs:
                q.append(x); s += x
                if len(q) > k: s -= q.pop(0)
                out.append(s / len(q))
            return out
        smooth = moving_avg(reward_hist, 20)[-1]

        resp_len_obs = float(torch.tensor([len(x.split()) for x in resp_obs]).float().mean().item())
        resp_len_pio = float(torch.tensor([len(x.split()) for x in resp_pio]).float().mean().item())

        wandb.log({
            "env/reward_raw_mean": float(raw_all.mean().item()),
            "env/reward_raw_std": float(raw_all.std().item()),
            "env/reward_mean_scaled": float(scaled_all.mean().item()),
            "env/reward_std_scaled": float(scaled_all.std().item()),
            "env/reward_smooth_scaled": smooth,
            "generation/resp_len_obs": resp_len_obs,
            "generation/resp_len_pio": resp_len_pio,
            "env/obs_valid_ratio": float(sum(1 for x in obs_valid if x is not None)) / max(1, len(obs_valid)),
            "env/pio_valid_ratio": float(sum(1 for x in pio_valid if x is not None)) / max(1, len(pio_valid)),
            "trainer/step": step,
            "env/reward_scaled_hist": wandb.Histogram(scaled_all.detach().cpu().numpy()),
        })
        wandb.log({f"ppo/{k}": v for k, v in (stats_obs or {}).items() if isinstance(v, (int, float))})
        wandb.log({f"ppo_b/{k}": v for k, v in (stats_pio or {}).items() if isinstance(v, (int, float))})

    print(">> Training finished.")
    out_dir = "./outputs/crea_bridge_last"
    ppo_trainer.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)

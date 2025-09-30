#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cory_withIRM_rag.py
- RAG 文脈付き Observer→Pioneer の2段 JSON 生成
- IRM（1..10を想定）→ 0..1線形正規化、または Z-score→tanh などでスケール
- TRL PPOTrainer に手動で rollouts を流して学習
"""

import os
import re
import json
import time
import math
import random
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
# tqdm の安全な import
try:
    from tqdm.auto import tqdm   # ✅ tqdm を関数として import
except ImportError:
    def tqdm(iterable=None, **kwargs):
        # tqdm が無ければフォールバック（そのまま iterable を返すだけ）
        return iterable if iterable is not None else range(kwargs.get("total", 0))


from typing import List, Tuple
import torch
import tyro
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.utils import is_bitsandbytes_available

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # bnb未インストールでも読み込めるように

# -----------------------------------------------------------------------------
# JSON utils（軽量修復・検証）
# -----------------------------------------------------------------------------
REQUIRED_KEYS = ["input_concepts", "new_concepts", "bridge_rationale", "plan", "risks", "title", "abstract"]

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

def extract_json_block(text: str) -> str | None:
    """最初に現れる最外カッコのJSONブロックを手動パースで抽出（正規表現に依存しない）"""
    if not text:
        return None
    # 先頭の '{' を探す（途中にプロンプトが混ざっていてもOK）
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    i = start
    in_str = False
    esc = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        i += 1
    return None  # 閉じカッコまで到達しなかった

def try_parse_json_relaxed(text: str):
    """プロンプト等が混ざった出力からJSON部分だけ抜いてパース。失敗時None"""
    jtxt = extract_json_block(text)
    if not jtxt:
        return None
    try:
        return json.loads(jtxt)
    except Exception:
        return None

def try_parse_json_relaxed(text: str):
    """プロンプト混入やゴミ付き出力からJSONだけ抜いてパース（失敗ならNone）"""
    jtxt = extract_json_block(text)
    if not jtxt:
        return None
    try:
        return json.loads(jtxt)
    except Exception:
        # ここで json_repair を使えるなら呼ぶ（導入していないならスキップ）
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

def _pretty_trunc(s: str, limit: int) -> str:
    if s is None:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [TRUNCATED {len(s)-limit} chars]"

def _print_samples(title: str, pairs: list[tuple[str,str]], max_chars: int = 1200):
    print(f"\n===== {title} =====")
    for i, (pr, rs) in enumerate(pairs):
        print(f"\n--- sample #{i} ---")
        print("[PROMPT]\n" + _pretty_trunc(pr, max_chars))
        print("\n[OUTPUT]\n" + _pretty_trunc(rs, max_chars))

def _strip_echo(prompts, outputs):
    cleaned = []
    for p, o in zip(prompts, outputs):
        if o.startswith(p):
            cleaned.append(o[len(p):].lstrip())
        else:
            cleaned.append(o)
    return cleaned



def make_irm_text_from_idea(idea: Dict[str, Any]) -> str:
    t = idea.get("title", "").strip()
    a = idea.get("abstract", "").strip()
    b = idea.get("bridge_rationale", "").strip()
    p = idea.get("plan", "").strip()
    return f"{t}\n\n{a}\n\nBridge: {b}\nPlan: {p}"

# === ここから追加: 「緩い」取り出し + フォールバック ===========================

def make_irm_text_from_idea_relaxed(idea: Dict[str, Any]) -> str:
    """キーが欠けていてもあるだけ拾ってIRM用テキストを作る（空にはしない）"""
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

_JSON_TITLE_RE = re.compile(r'"title"\s*:\s*"([^"]+)"', re.S)
_JSON_ABS_RE   = re.compile(r'"abstract"\s*:\s*"([^"]+)"', re.S)

def fallback_irm_text_from_raw_response(resp_text: str, seed_problem: str = "") -> str:
    """JSON抽出に失敗したときのIRM入力テキスト生成（空回避）"""
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
    body = s.strip()[:1600]  # 過長防止
    head = seed_problem.strip() or "Generated Idea"
    return f"{head}\n\n{body}"

def _preview(xs: List[str], n=4):
    out = []
    for x in xs[:n]:
        h = (x[:140] + '…') if len(x) > 140 else x
        out.append((len(x), h))
    return out

# -----------------------------------------------------------------------------
# IRM（単一スコア）: 1..10 → 0..1 に正規化
# -----------------------------------------------------------------------------
class IRMReward:
    """
    Idea Reward Model wrapper (Transformers)
    - 量子化: quantization in {"none","8bit","4bit"} をサポート（推奨: 8bit/4bit は BitsAndBytesConfig + device_map="auto"）
    - 量子化時は .to(cuda) を呼ばない（Accelerate が層ごとに自動割当）
    - 非量子化時のみ明示 .to(device)
    - 出力: raw（~1..10想定）, norm01=(raw-1)/9 を [0,1] に clamp
    """

    def __init__(
        self,
        irm_model_dir: str,
        max_length: int = 512,
        device: torch.device | None = None,
        quantization: str = "none",                 # "none" | "8bit" | "4bit"
        torch_dtype: str | None = "auto",           # "auto" | "float16" | "bfloat16" | "float32"
        device_map: str = "auto",                   # 量子化時は基本 "auto" を推奨
        low_cpu_mem_usage: bool = True,             # 初期ロード時のメモリ節約
        offload_folder: str | None = None,          # CPU/disk offload を使う場合の一時フォルダ
        max_memory: dict | None = None,             # 例: {"cuda:0":"19GiB","cpu":"64GiB"}
    ):
        self.max_length = max_length
        self._uses_device_map = False  # 量子化+device_map=auto の時 True
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # tokenizer
        self.tok = AutoTokenizer.from_pretrained(irm_model_dir, use_fast=True)

        # dtype 解決
        if torch_dtype == "float16":
            _dtype = torch.float16
        elif torch_dtype == "bfloat16":
            _dtype = torch.bfloat16
        elif torch_dtype == "float32":
            _dtype = torch.float32
        else:
            _dtype = None  # auto

        def _maybe_dtype(kw: dict):
            if _dtype is not None and "torch_dtype" not in kw:
                kw["torch_dtype"] = _dtype
            return kw

        # ===== 量子化経路 =====
        if quantization in ("8bit", "4bit"):
            if not is_bitsandbytes_available() or BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes が見つかりません。`pip install -U bitsandbytes` を実行してください。"
                )
            load_in_8bit = (quantization == "8bit")
            load_in_4bit = (quantization == "4bit")
            bnb_cfg = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                llm_int8_threshold=6.0,        # 保守的設定
                llm_int8_has_fp16_weight=False # 分岐抑制
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
            _maybe_dtype(kw)  # dtype を指定したい場合のみ付与（bnbでも有効）

            # モデルをロード（※ .to() は呼ばない）
            self.model = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, **kw)
            self._uses_device_map = True
            self.model.eval()
            return

        # ===== 非量子化（フル精度 or 半精度） =====
        kw = {"low_cpu_mem_usage": low_cpu_mem_usage}
        if max_memory is not None:
            kw["max_memory"] = max_memory
        _maybe_dtype(kw)
        try:
            # CUDA が使えれば CUDA へ、ダメなら CPU
            if torch.cuda.is_available():
                m = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, **kw)
                m.to(self.device).eval()
                self.model = m
            else:
                self.device = torch.device("cpu")
                m = AutoModelForSequenceClassification.from_pretrained(irm_model_dir, **kw)
                m.to(self.device).eval()
                self.model = m
        except RuntimeError:
            # OOM 等は CPU にフォールバック
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
        # 量子化+device_map=auto の場合、入力は CPU のままで OK（Accelerate が自動でルーティング）
        if not self._uses_device_map:
            enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits.squeeze(-1).detach().float()  # (B,)
        raw = logits
        norm01 = torch.clamp((raw - 1.0) / 9.0, 0.0, 1.0)
        return raw, norm01

# -----------------------------------------------------------------------------
# 移動平均 / EMA スケール
# -----------------------------------------------------------------------------
def moving_avg(xs: List[float], k: int = 10) -> List[float]:
    out, s, q = [], 0.0, []
    for x in xs:
        q.append(x); s += x
        if len(q) > k: s -= q.pop(0)
        out.append(s / len(q))
    return out

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

# -----------------------------------------------------------------------------
# 設定
# -----------------------------------------------------------------------------
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
    group: Optional[str] = "cory-json-irm-rag"
    run_name: Optional[str] = f"cory-json-irm-rag-{int(time.time())}"

    # IRM
    irm_model_dir: str = "./IRM/irm_iclr_model"
    irm_max_len: int = 512
    reward_scale_mode: str = "raw_zscore_tanh"  # "norm01_identity" にすると切り分けしやすい
    reward_scale_beta: float = 1.5

    # seeds
    seeds_path: str = "./data/research_seeds.jsonl"
    max_seeds: int = 100000

    # IRM quantizatrion
    irm_quantization: str = "8bit"     # "none" | "8bit" | "4bit"
    irm_torch_dtype: str = "auto"      # "auto" | "float16" | "bfloat16" | "float32"
    irm_device_map: str = "auto"
    irm_low_cpu_mem_usage: bool = True
    irm_offload_folder: str | None = None
    irm_max_memory: dict | None = None

    print_every_steps: int = 100        # 何ステップごとに印字するか（0なら無効）
    print_n_samples: int = 2            # 1回の印字で表示するサンプル数（バッチ先頭から）
    print_prompt_max_chars: int = 1200  # 端末表示でのプロンプト/出力の最大文字数
    print_ctx_docs: int = 3             # 印字時のObserver用RAG文脈の最大件数（見やすさ用）

# -----------------------------------------------------------------------------
# データ（RAGシード）
# -----------------------------------------------------------------------------
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
            ))
            if limit is not None and len(seeds) >= limit:
                break
    assert len(seeds) > 0, f"no seeds at {path}"
    return seeds

# -----------------------------------------------------------------------------
# プロンプト（RAG文脈を注入）
# -----------------------------------------------------------------------------
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

# 既存の OBSERVER_JSON_PROMPT を置き換え
OBSERVER_JSON_PROMPT = (
    "You are a creative research ideator.\n"
    "Given the TOPIC, PROBLEM, and SOURCE DOMAINS, propose ONE research idea.\n"
    "Respond ONLY in valid JSON matching the schema.\n\n"
    "TOPIC: {topic}\n"
    "PROBLEM: {problem}\n"
    "SOURCES: {sources}\n"
    "CONSTRAINTS: {constraints}\n\n"
    "{input_hint}"
    "Schema (keys required): {{\"input_concepts\":[\"...\"], \"new_concepts\":[\"...\"], "
    "\"bridge_rationale\":\"...\", \"plan\":\"...\", \"risks\":[\"...\"], "
    "\"title\":\"...\", \"abstract\":\"...\"}}\n\n"
    "Example (very short):\n"
    "{{\n"
    "  \"input_concepts\": [\"vision\"],\n"
    "  \"new_concepts\": [\"robotics\"],\n"
    "  \"bridge_rationale\": \"Combine visual grounding with robotic planning.\",\n"
    "  \"plan\": \"Link perception to action with a small prototype pipeline.\",\n"
    "  \"risks\": [\"data scarcity\"],\n"
    "  \"title\": \"Vision-Guided Robotic Planning\",\n"
    "  \"abstract\": \"Connect computer vision to robot task planning with a minimal pipeline.\"\n"
    "}}\n\n"
    "Requirements: encourage bisociation (combine distant concepts).\n"
    "Return JSON only. No prose outside JSON.\n\n"
    "{rag_block}"
)

# 既存の PIONEER_REFINE_JSON_PROMPT を置き換え
PIONEER_REFINE_JSON_PROMPT = (
    "You are a research editor improving feasibility and clarity.\n"
    "Refine the JSON idea to reduce risks and increase feasibility, while preserving novelty.\n"
    "Return JSON only. No extra text.\n\n"
    "Example (very short):\n"
    "{\n"
    "  \"input_concepts\": [\"vision\"],\n"
    "  \"new_concepts\": [\"robotics\"],\n"
    "  \"bridge_rationale\": \"...\",\n"
    "  \"plan\": \"...\",\n"
    "  \"risks\": [\"...\"],\n"
    "  \"title\": \"...\",\n"
    "  \"abstract\": \"...\"\n"
    "}\n\n"
    "{idea_json}\n\n{rag_block}"
)


def build_observer_prompt(seed: SeedItem, ctx_docs: int = 5) -> str:
    rag_block = build_ctx_block(seed.__dict__, max_docs=ctx_docs)
    ics = (seed.input_concepts or [])
    input_hint = f"INPUT_CONCEPTS_HINT: {', '.join(ics)}\n" if ics else ""
    return OBSERVER_JSON_PROMPT.format(
        topic=seed.topic, problem=seed.problem, sources=", ".join(seed.sources),
        constraints=seed.constraints or "", input_hint=input_hint, rag_block=rag_block
    )

def build_pioneer_prompt(observer_json: Dict[str, Any], seed: SeedItem, ctx_docs: int = 5) -> str:
    rag_block = build_ctx_block(seed.__dict__, max_docs=ctx_docs)
    idea_json = json.dumps(observer_json, ensure_ascii=False)
    return PIONEER_REFINE_JSON_PROMPT.format(idea_json=idea_json, rag_block=rag_block)

def build_irm_reward(args, device) -> IRMReward:
    """
    args から IRMReward を安全に初期化するヘルパ（tyro想定）
    - 推奨: args に以下の属性を持たせる
        irm_model_dir: str
        irm_max_len: int = 512
        irm_quantization: str = "8bit"  # "none" | "8bit" | "4bit"
        irm_torch_dtype: str = "auto"   # "auto" | "float16" | "bfloat16" | "float32"
        irm_device_map: str = "auto"    # 量子化時 "auto" 推奨
        irm_low_cpu_mem_usage: bool = True
        irm_offload_folder: Optional[str] = None
        irm_max_memory: Optional[dict] = None  # {"cuda:0":"19GiB","cpu":"64GiB"} など
    """
    q = getattr(args, "irm_quantization", "none")
    if q not in ("none", "8bit", "4bit"):
        q = "none"

    return IRMReward(
        irm_model_dir=args.irm_model_dir,
        max_length=getattr(args, "irm_max_len", 512),
        device=device,
        quantization=q,
        torch_dtype=getattr(args, "irm_torch_dtype", "auto"),
        device_map=getattr(args, "irm_device_map", "auto"),
        low_cpu_mem_usage=getattr(args, "irm_low_cpu_mem_usage", True),
        offload_folder=getattr(args, "irm_offload_folder", None),
        max_memory=getattr(args, "irm_max_memory", None),
    )


# -----------------------------------------------------------------------------
# 生成ヘルパ
# -----------------------------------------------------------------------------
@dataclass
class GenCfg:
    min_new_tokens: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    no_repeat_ngram_size: int = 3     # ← 追加
    do_sample: bool = True            # ← 追加

def generate_texts(ppo_trainer, prompts: List[str], cfg: GenCfg) -> List[str]:
    out = ppo_trainer.generate(
        prompts,
        do_sample=True,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        temperature=cfg.temperature,
        repetition_penalty=cfg.repetition_penalty,
        no_repeat_ngram_size=cfg.no_repeat_ngram_size,  # ← 追加
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

# -----------------------------------------------------------------------------
# 報酬スケール
# -----------------------------------------------------------------------------
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
        return [ri.detach() for ri in r]

    return scale_rewards

# -----------------------------------------------------------------------------
# メイン
# -----------------------------------------------------------------------------
def main(args: TrainArgs):
    wandb.init(project=args.group or "cory-json-irm-rag", name=args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PPOTrainer 構築
    from trainer_builder import build_ppo_trainer
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
            "target_modules": ["c_attn", "c_proj", "c_fc"],
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
    irm_reward = build_irm_reward(args, device=ppo_trainer.accelerator.device)
    scale_rewards = build_reward_scaler(args)
    seeds = load_seeds(args.seeds_path, limit=args.max_seeds)

    # 生成設定
    cfg_obs = GenCfg(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=0.4,             # ← 0.7 から下げる
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=1.1,      # ← 1.0 → 1.1
        no_repeat_ngram_size=4,      # ← 追加
        do_sample=True               # ← 追加
    )
    cfg_pio = GenCfg(
        min_new_tokens=128,
        max_new_tokens=min(700, args.max_new_tokens),
        temperature=0.4,             # ← 0.7 → 0.4
        top_p=0.9,
        top_k=args.top_k,
        repetition_penalty=1.1,      # ← 1.05 → 1.1
        no_repeat_ngram_size=4,      # ← 追加
        do_sample=True               # ← 追加
    )

    reward_hist: List[float] = []

    # === エポック風のロバストなバッチイテレータ（同一バッチ内は重複なし） ===
    def batch_iterator(data, batch_size):
        i = 0
        pool = data[:]
        random.shuffle(pool)
        while True:
            if len(pool) == 0:
                raise RuntimeError("no seeds")
            if batch_size > len(pool):
                # シードが少ない時は重複ありフォールバック（学習続行を優先）
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
        resp_obs = _strip_echo(obs_prompts, resp_obs)  # プロンプトのエコーを除去
        obs_jsons = [try_parse_json_relaxed(t) for t in resp_obs]  # ゴミ混入でもJSON部だけ抽出
        obs_valid = [x if (x and validate_schema(x)) else None for x in obs_jsons]

        # --- Pioneer（Observer失敗サンプルはスキップ） ---
        pio_prompts = []
        mask_do_pio = []
        for s, idea in zip(batch, obs_valid):
            if idea is None:
                mask_do_pio.append(False)
                pio_prompts.append(None)
            else:
                mask_do_pio.append(True)
                pio_prompts.append(build_pioneer_prompt(idea, s, ctx_docs=5))

        # 実際に生成するのは有効サンプルだけ
        indices = [i for i, ok in enumerate(mask_do_pio) if ok]
        resp_pio = [None] * len(batch)
        if indices:
            sub_prompts = [pio_prompts[i] for i in indices]
            sub_resp = generate_texts(ppo_trainer, sub_prompts, cfg_pio)
            sub_resp = _strip_echo(sub_prompts, sub_resp)
            for i, r in zip(indices, sub_resp):
                resp_pio[i] = r

        # 以降の処理で落ちないよう、None→空文字に正規化
        resp_pio = [r if isinstance(r, str) else "" for r in resp_pio]

        # PioneerのJSONパース（緩和版）
        pio_jsons = [try_parse_json_relaxed(t) for t in resp_pio]
        pio_valid = [x if (x and validate_schema(x)) else None for x in pio_jsons]
        
        if args.print_every_steps and (step % args.print_every_steps == 0):
            obs_pairs = list(zip(obs_prompts, resp_obs))[: max(1, args.print_n_samples)]
            pio_pairs = list(zip(pio_prompts, resp_pio))[: max(1, args.print_n_samples)]

            print(f"\n\n######## EVAL SNAPSHOT @ step {step} ########")
            _print_samples("OBSERVER (idea proposal JSON)", obs_pairs, args.print_prompt_max_chars)
            _print_samples("PIONEER (refine JSON)", pio_pairs, args.print_prompt_max_chars)
            print("######## END SNAPSHOT ########\n")

        # --- IRM scores（空文字禁止：緩い整形＋フォールバックで必ず非空に） ---
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

        # デバッグ: 入力サンプルをW&Bに出す（長さ＆先頭断片）
        try:
            wandb.log({
                "debug/idea_obs_samples": wandb.Table(columns=["len","head"], data=_preview(idea_obs_texts, 4)),
                "debug/idea_pio_samples": wandb.Table(columns=["len","head"], data=_preview(idea_pio_texts, 4)),
            })
        except Exception:
            pass

        raw_obs_t, n01_obs_t = irm_reward.score(idea_obs_texts)
        raw_pio_t, n01_pio_t = irm_reward.score(idea_pio_texts)

        # --- PPO updates --- の直前あたり（raw_obs_t, n01_obs_t 取得後）
        r_obs = scale_rewards(raw_obs_t, n01_obs_t)
        # JSON妥当性シェーピング（例）
        for i, valid in enumerate(obs_valid):
            r_obs[i] = r_obs[i] + (0.2 if valid is not None else -0.5)

        stats_obs = {}
        stats_pio = {}

        try:
            # ★ 位置引数で渡す（queries, responses, rewards）
            stats_obs = ppo_trainer.step(obs_prompts, resp_obs, r_obs)
        except Exception:
            pass

        if (step % max(1, args.swap_every)) == 0:
            try:
                r_pio = scale_rewards(raw_pio_t, n01_pio_t)
                for i, valid in enumerate(pio_valid):
                    r_pio[i] = r_pio[i] + (0.2 if valid is not None else -0.5)

                # ★ こちらも位置引数
                stats_pio = ppo_trainer.step(pio_prompts, resp_pio, r_pio)
            except Exception:
                pass

        # --- logs ---
        raw_all = torch.cat([raw_obs_t, raw_pio_t], dim=0)
        n01_all = torch.cat([n01_obs_t, n01_pio_t], dim=0)
        scaled_all = torch.stack(scale_rewards(raw_all, n01_all))
        reward_hist.append(float(scaled_all.mean().item()))
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

        # ---- optional: デバッグ（最初の100ステップだけ） ----
        # if step < 100:
        #     print(f"[step {step}] IRM raw obs mean/std/min/max:",
        #           float(raw_obs_t.mean()), float(raw_obs_t.std()),
        #           float(raw_obs_t.min()),  float(raw_obs_t.max()))
        #     print(f"[step {step}] IRM raw pio mean/std/min/max:",
        #           float(raw_pio_t.mean()), float(raw_pio_t.std()),
        #           float(raw_pio_t.min()),  float(raw_pio_t.max()))
        #     print(f"[step {step}] scaled mean/std:",
        #           float(scaled_all.mean().item()),
        #           float(scaled_all.std().item()))

    print(">> Training finished.")
    ppo_trainer.save_pretrained("./outputs/cory_json_irm_rag_last")
    tokenizer.save_pretrained("./outputs/cory_json_irm_rag_last")


if __name__ == "__main__":
    args = tyro.cli(TrainArgs)
    main(args)

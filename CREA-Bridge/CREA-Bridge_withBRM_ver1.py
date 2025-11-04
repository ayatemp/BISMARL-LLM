# -*- coding: utf-8 -*-
"""
CREA-Bridge (Dual-GPU: policy=CUDA:0, IRM=CUDA:1)

- Bridger → Pioneer → Observer の3段生成 + IRMスコア + PPO更新（バッチ対応）
- 出力は "論文タイトル + アブスト" 形式に統一（IRM 分布合わせ）
- 重要: 報酬は「policy の応答テキスト」から算出（信用割当の欠落を修正）
- str.format 廃止 → [[...]] 置換で安全化
- 動的バッチ調整（eff_bsz に合わせて trainer.config.batch_size/mini_batch_size を上書き）
- IRM: 生logitsヒスト、Platt/μσキャリブ、穏当なフォールバック（ztemp）
- W&B: reward系統計、logitsヒスト、samples/outputs テーブル（rawも保持）
- TorchInductor/Triton を環境変数で無効化（triton未導入環境の安定化）
- BRIDGER: 未完JSON自動修復 + "keys":[...] 直抜きフォールバック（サルベージ強化）




export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_TRITON_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
# 8bit/4bitを使う場合は、既に導入した FORCE_QUANT を併用してOK

python CREA-Bridge_withBRM.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --irm-model-dir ayarnte/IRM_high_ver \
  --seeds-path CORY_withRAG/data/research_seeds.fixed.jsonl \
  --max-seeds 20000 \
  --total-steps 200 \
  --ppo-epochs 1 \
  --batch-size 4 --mini-batch-size 2 \
  --learning-rate 6e-6 \
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.05 \
  --irm-device cuda:1 \
  --reward-mode calib \
  --max-new-tokens 64 \
  --log-every-n 10 \
  --bridge-weight 0.15 \
  --bridge-emb-model sentence-transformers/all-MiniLM-L6-v2 \
  --bridge-topk 3

python CREA-Bridge_withBRM.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --irm-model-dir ayarnte/IRM_high_ver \
  --seeds-path CORY_withRAG/data/research_seeds.fixed.jsonl \
  --max-seeds 20000 \
  --total-steps 200 \
  --ppo-epochs 1 \
  --batch-size 4 --mini-batch-size 2 \
  --learning-rate 6e-6 \
  --lora-r 16 --lora-alpha 16 --lora-dropout 0.05 \
  --irm-device cuda:1 \
  --reward-mode calib \
  --max-new-tokens 64 \
  --log-every-n 1 \
  --bridge-weight 0.15 \
  --bridge-emb-model sentence-transformers/all-MiniLM-L6-v2 \
  --bridge-topk 3
"""

import os
import sys
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    StoppingCriteria, StoppingCriteriaList
)
from trl import AutoModelForCausalLMWithValueHead, PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model

# ====== Env guards (Inductor/Triton off) ======
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("PYTORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
if "TORCH_LOGS" in os.environ:
    os.environ.pop("TORCH_LOGS", None)

try:
    import importlib
    importlib.import_module("triton.ops")
    _HAS_TRITON_OPS = True
except Exception:
    _HAS_TRITON_OPS = False
    print("[WARN] TorchInductor/Triton ops not found; proceeding without them.")

# ====== Optional deps ======
try:
    from tqdm import trange
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from transformers import AutoModel, AutoTokenizer  # 既に import ありなら重複OK
import torch.nn.functional as F

# ====== Constants ======
MAX_QUERY_TOKENS_FOR_PPO = 256
MAX_RESP_TOKENS_FOR_PPO  = 128

# ====== Utils ======
def _normalize_ideas_obj(obj) -> Dict[str, Any]:
    """
    obj が dict のとき: {"ideas": [...], "rationale": "..."} を期待
    obj が list のとき: それ自体を ideas とみなし、rationale は空
    その他: 空で返す
    返り値: {"ideas": List[Dict], "rationale": str}
    """
    ideas, rationale = [], ""
    if isinstance(obj, dict):
        ideas = obj.get("ideas") or []
        rationale = obj.get("rationale") or ""
    elif isinstance(obj, list):
        # list of dicts を ideas とみなす（トップレベル配列パターン救済）
        ideas = obj
    # 正規化
    if isinstance(ideas, dict):
        ideas = [ideas]
    if not isinstance(ideas, list):
        ideas = []
    # dict 以外の要素は弾く
    ideas = [it for it in ideas if isinstance(it, dict)]
    return {"ideas": ideas, "rationale": rationale}

def assert_cuda_device(dev_str: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Please run on a GPU machine.")
    d = torch.device(dev_str)
    if d.type != "cuda" or d.index is None:
        raise RuntimeError(f"{dev_str} is not a valid CUDA device (e.g., 'cuda:0').")
    return d

def set_default_dtype_bf16_if_available():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)

def _preferred_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

class EMA:
    def __init__(self, beta=0.9):
        self.beta = beta
        self.value = None
    def update(self, x: float) -> float:
        if x is None or (isinstance(x, float) and (x != x)):
            return self.value if self.value is not None else float("nan")
        self.value = x if self.value is None else (self.beta*self.value + (1-self.beta)*x)
        return self.value

SMART_QUOTES = {"“": '"', "”": '"', "„": '"', "‟": '"', "’": "'", "‘": "'"}
def _desmart(s: str) -> str:
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s

def _strip_fences(s: str) -> str:
    # ```lang\n ... \n``` / ~~~lang\n ... \n~~~ を安全に剥がす
    s = re.sub(r"^```[\w-]*\s*", "", s.strip(), flags=re.DOTALL)
    s = re.sub(r"```$", "", s.strip(), flags=re.DOTALL)
    s = re.sub(r"^~~~[\w-]*\s*", "", s.strip(), flags=re.DOTALL)
    s = re.sub(r"~~~$", "", s.strip(), flags=re.DOTALL)
    return s

def extract_first_json(text: str):
    if not text:
        return None
    s = _desmart(_strip_fences(text))
    cands = []
    cands += [m.group(0) for m in re.finditer(r"\{.*?\}", s, flags=re.DOTALL)]
    cands += [m.group(0) for m in re.finditer(r"\[.*?\]", s, flags=re.DOTALL)]
    def _try(js):
        js = re.sub(r",\s*([}\]])", r"\1", js)
        try: return json.loads(js)
        except: return None
    parsed = []
    for js in cands:
        obj = _try(js)
        if obj is not None:
            parsed.append(obj)
    if not parsed:
        return None
    # Prefer dict with keys-like fields
    for obj in parsed:
        if isinstance(obj, dict):
            kk = {str(k).lower(): v for k,v in obj.items()}
            if ("keys" in kk) or ("key" in kk) or ("ideas" in kk) or ("bisociation_keys" in kk):
                return obj
    return parsed[0]

def safe_get(d, key, default=""):
    try:
        if isinstance(d, dict):
            return d.get(key, default)
    except:
        pass
    return default

# ==== JSON 修復 & keys 直抜き（強化サルベージ） ====
def _brace_depth(s: str) -> int:
    d = 0
    for ch in s:
        if ch == "{": d += 1
        elif ch == "}": d -= 1
    return d

def _repair_minified_json(s: str) -> str:
    s = _desmart(_strip_fences(s)).strip()
    if not s: return s
    if not (s.lstrip().startswith("{") or s.lstrip().startswith("[")): return s
    depth = _brace_depth(s)
    if depth > 0: s = s + ("}" * depth)
    s = re.sub(r",\s*([}\]])", r"\1", s)  # 末尾カンマ除去
    if not s.endswith("}") and not s.endswith("]"):
        s = s + "}"
    return s

_KEYLIST_RE = re.compile(r'"keys"\s*:\s*\[(.*?)\]', flags=re.DOTALL | re.IGNORECASE)

def _salvage_keys_from_any(text: str) -> list:
    if not text: return []
    s = _desmart(_strip_fences(text))
    m = _KEYLIST_RE.search(s)
    if not m:
        return _salvage_keys_from_text(text)
    payload = m.group(1)
    toks = []
    for frag in payload.split(","):
        frag = frag.strip()
        if not frag: continue
        frag = re.sub(r'^[\'"]|[\'"]$', '', frag).strip()
        if frag: toks.append(frag)
    uniq, seen = [], set()
    for k in toks:
        k2 = k[:64]; kl = k2.lower()
        if kl and kl not in seen and len(k2.split()) <= 8:
            uniq.append(k2); seen.add(kl)
    return uniq[:9]

# ===== Seeds =====
class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]): self.prompts = prompts
    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx): return {"prompt": self.prompts[idx]}

def _mk_task_from_seed(obj: Dict[str, Any]) -> str:
    topic = str(obj.get("topic","")).strip()
    problem = str(obj.get("problem","")).strip()
    constraints = str(obj.get("constraints","")).strip()
    src = obj.get("sources", [])
    if isinstance(src, str):
        sources = [x.strip() for x in src.split(",") if x.strip()]
    elif isinstance(src, list):
        sources = [str(x).strip() for x in src if str(x).strip()]
    else:
        sources = []
    ctx = obj.get("context", [])
    ctx_summ = []
    if isinstance(ctx, list):
        for it in ctx[:3]:
            try:
                t = it.get("title","")
                a = it.get("abstract","")
                if t:
                    ctx_summ.append(f"{t}: {a[:160]}")
            except:
                pass
    lines = []
    if topic: lines.append(f"TOPIC: {topic}")
    if problem: lines.append(f"PROBLEM: {problem}")
    if sources: lines.append("SOURCES: " + ", ".join(sources[:8]))
    if constraints: lines.append(f"CONSTRAINTS: {constraints}")
    if ctx_summ: lines.append("CONTEXT: " + " | ".join(ctx_summ))
    return "\n".join(lines) if lines else (topic or "Research ideation")

def load_prompts_from_jsonl(path: str, max_items: Optional[int]) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items: break
            s = line.strip()
            if not s: continue
            try:
                obj = json.loads(s)
                prompts.append(_mk_task_from_seed(obj))
            except:
                prompts.append(s)
    return prompts

def gen_bridger_keys(bundle, task: str, n: int = 2):
    cand_lists = []
    raw_outs = []  # ← 追加
    for i in range(max(1, n)):
        bridger_prompt = _fill(BRIDGER_PROMPT, {"TASK": task})
        out_i = run_generate_cuda(
            bundle.model, bundle.tokenizer, bridger_prompt,
            GEN_BRIDGER, stopper=JSONKeysStopper(bundle.tokenizer)
        )
        raw_outs.append(out_i)  # ← 追加

        repaired = _repair_minified_json(out_i)
        bj = extract_first_json(repaired) or extract_first_json(out_i)
        try:
            kj = _enforce_keys_only_json(bj) if bj is not None else {"keys": []}
        except Exception:
            kj = {"keys": []}
        if not kj.get("keys"):
            kj["keys"] = _salvage_keys_from_any(out_i)
        cand_lists.append(kj.get("keys", []))

    # マージ（既存のまま）
    uniq, seen = [], set()
    for ks in cand_lists:
        for k in ks:
            k2 = (k or "").strip()[:64]
            low = k2.lower()
            if not k2 or low in seen: continue
            if 1 <= len(k2.split()) <= 8:
                uniq.append(k2); seen.add(low)
    if not uniq:
        rough = [w for w in re.split(r"[\s,/|]+", str(task)) if len(w) > 3][:6]
        uniq = rough or ["LLM","Robotics","Grounding","Multimodal","Evaluation"]

    return uniq[:9], raw_outs  # ← keys と raw を返す

# ===== Prompts (academic style) =====
BRIDGER_PROMPT = (
    "You are BRIDGER for research bisociation.\n"
    "Output STRICTLY ONE minified JSON object and NOTHING ELSE.\n"
    "Your very first character MUST be '{' and the last character MUST be '}'.\n"
    'Schema: {\"keys\": [\"<short bisociative key>\", \"...\"]}\n'
    "Rules:\n"
    "- ONLY the field 'keys' is allowed; no extra fields.\n"
    "- 5 to 9 concise keys, each <= 6 words.\n"
    "- Avoid duplicates; keep technical and actionable.\n"
    "TASK:\n[[TASK]]\n"
)

PIONEER_PROMPT = (
    "You are PIONEER. Propose 1-3 concrete research ideas in academic paper form.\n"
    'Return JSON with schema: {\"ideas\":[{\"title\":\"...\",\"abstract\":\"...\"}],\"rationale\":\"...\"}\n'
    "Rules:\n"
    "- 'title' must be a concise academic paper title.\n"
    "- 'abstract' MUST be one paragraph (no line breaks) and follow:\n"
    "  (1) Motivation, (2) Proposed Method (specific), (3) Experimental Setup (dataset, model size, metrics, baselines), (4) Expected Contribution.\n"
    "- Use formal academic tone, no bullet points.\n\n"
    "TASK:\n[[TASK]]\n\nBRIDGER_KEYS:\n[[KEYS]]\n"
    "Output STRICTLY minified JSON."
)

OBSERVER_PROMPT = (
    "You are OBSERVER. Improve the ideas for clarity, originality, feasibility, and academic writing quality.\n"
    "Return JSON with the SAME SCHEMA.\n"
    "Rewrite abstracts to match real conference-style abstracts.\n"
    "Rules:\n"
    "- One paragraph abstract (no line breaks), academic tone.\n"
    "- Include dataset names, evaluation metrics (e.g., BLEU, FID, accuracy), and ablation plans where applicable.\n"
    "- Make the novelty/contribution explicit.\n\n"
    "TASK:\n[[TASK]]\n\nPIONEER_JSON:\n[[PIONEER_JSON]]\n"
    "Output STRICTLY minified JSON."
)

# ===== Generation =====
@dataclass
class GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 0

# BRIDGERは貪欲 + 少し長め
GEN_BRIDGER = GenCfg(max_new_tokens=80,  temperature=0.25, top_p=1.0, top_k=0)
GEN_PIONEER = GenCfg(max_new_tokens=96,  temperature=0.2, top_p=0.9, top_k=0)
GEN_OBSERVER = GenCfg(max_new_tokens=96,  temperature=0.1, top_p=0.9, top_k=0)

class JSONKeysStopper(StoppingCriteria):
    def __init__(self, tokenizer, lookback_chars: int = 2048):
        self.tok = tokenizer
        self.lookback = lookback_chars
        self.started = False
        self.depth = 0
        self.seen_open = False
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        text = self.tok.decode(input_ids[0][-1024:], skip_special_tokens=True)
        if not self.started and re.search(r'\"keys\"', text):
            self.started = True
        for ch in text[-256:]:
            if ch == "{":
                self.depth += 1; self.seen_open = True
            elif ch == "}":
                self.depth -= 1
        return (self.started and self.seen_open and self.depth <= 0)

def run_generate_cuda(model, tokenizer, prompt: str, gen_cfg: GenCfg, stopper: Optional[StoppingCriteria]=None) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to("cuda:0") for k, v in enc.items()}
    stopping = StoppingCriteriaList([stopper]) if stopper is not None else None

    # ← 修正点：temperature > 0 なら常に do_sample=True
    do_sample = (gen_cfg.temperature is not None) and (gen_cfg.temperature > 0.0)

    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=do_sample,
            temperature=(gen_cfg.temperature if do_sample else None),
            top_p=(gen_cfg.top_p if do_sample else None),
            top_k=(gen_cfg.top_k if do_sample else None),
            max_new_tokens=gen_cfg.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            stopping_criteria=stopping,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

# ===== IRM =====
class IRMScorer:
    """
    - score_text(): 報酬として使う確率（校正済み）を返す
    - logits_text(): 生logitsを返す
    - calibrate_*(): mu/sigma/Platt を設定（--irm-calib-path の内容をロードする場合は load_calib()）
    """
    def __init__(self, model_dir: str, max_len: int = 512, device: str = "cuda:1"):
        dev = assert_cuda_device(device)
        self.device = str(dev)
        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.max_len = max_len
        # キャリブ係数
        self.mu = 0.0
        self.sigma = 1.0
        self.platt_coef = None  # (a, b) for sigmoid(a*logit + b)

    def load_calib(self, path: Optional[str]):
        if not path: return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            self.mu = float(obj.get("mu", 0.0))
            self.sigma = float(obj.get("sigma", 1.0))
            pl = obj.get("platt", None)
            if isinstance(pl, dict) and "coef" in pl and "intercept" in pl:
                self.platt_coef = (float(pl["coef"]), float(pl["intercept"]))
            print(f"[CALIB] Loaded: mu={self.mu:.4f}, sigma={self.sigma:.4f}, platt={self.platt_coef}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to load calib from {path}: {e}")
            return False

    @torch.no_grad()
    def logits_text(self, title: str, abstract: str) -> torch.Tensor:
        text = f"Title: {title}\nAbstract: {abstract}"
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        return out.logits.squeeze(0).detach()

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _platt_prob(self, logit_val: float) -> float:
        if self.platt_coef is None:
            return self._sigmoid(logit_val)
        a, b = self.platt_coef
        return self._sigmoid(a * logit_val + b)

    @torch.no_grad()
    def score_text(self, title: str, abstract: str,
                   reward_mode: str = "prob",
                   reward_temp: float = 4.0) -> float:
        """
        reward_mode:
          - "prob": softmax最大/二値sigmoid（素の確率）
          - "logit": 生logitを直接（>0 大、<0 小）
          - "ztemp": z正規化して sigmoid(T*z)
          - "platt": Platt（a*logit+b→sigmoid）
          - "calib": platt があれば platt、なければ ztemp
        """
        logits = self.logits_text(title, abstract)
        if logits.ndim == 0 or logits.numel() == 1:
            lg = float(logits.item())
        else:
            lg = float(torch.max(logits).item())

        if reward_mode == "logit":
            return lg
        elif reward_mode == "ztemp":
            z = (lg - self.mu) / (self.sigma + 1e-6)
            return self._sigmoid(z * reward_temp)
        elif reward_mode == "platt":
            return self._platt_prob(lg)
        elif reward_mode == "calib":
            if self.platt_coef is not None:
                return self._platt_prob(lg)
            else:
                z = (lg - self.mu) / (self.sigma + 1e-6)
                return self._sigmoid(z * reward_temp)
        else:  # "prob"
            return self._sigmoid(lg)

class BridgeScorer:
    """
    R_bridge = mean(topK_i [ sim(key_i, idea) * (1 - sim(key_i, task)) ])
    - すべて cosine。安全のため cos∈[-1,1] を [0,1] に線形マップして使用。
    - 埋め込みは同一エンコーダで統一。CUDA:1 で動かす（デフォルトは IRM と同じGPU）。
    - 簡易キャッシュで再計算を削減。
    """
    def __init__(self, model_name: str, device: str):
        self.device = str(assert_cuda_device(device))
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.enc = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.cache = {}  # text -> tensor

    @torch.no_grad()
    def _embed(self, text: str) -> torch.Tensor:
        key = ("e", text[:4096])  # ハッシュの代用
        if key in self.cache:
            return self.cache[key]
        batch = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.enc(**batch)
        if hasattr(out, "last_hidden_state"):
            h = out.last_hidden_state  # [1, L, H]
            e = (h.masked_fill(~batch["attention_mask"].unsqueeze(-1).bool(), 0.0).sum(dim=1) /
                 batch["attention_mask"].sum(dim=1, keepdim=True).clamp_min(1)).squeeze(0)
        else:
            e = out.pooler_output.squeeze(0)
        e = F.normalize(e, dim=-1)
        self.cache[key] = e
        return e

    @staticmethod
    def _cos01(a: torch.Tensor, b: torch.Tensor) -> float:
        # cosine in [-1,1] -> [0,1]
        c = F.cosine_similarity(a, b, dim=-1).item()
        return 0.5 * (c + 1.0)

    @torch.no_grad()
    def score(self, task_text: str, keys: list[str], idea_text: str, topk: int = 3) -> float:
        if not keys:
            return 0.0
        e_task = self._embed(task_text)
        e_idea = self._embed(idea_text)

        vals = []
        for k in keys:
            if not k or not isinstance(k, str):
                continue
            e_key = self._embed(k)
            sim_key_idea = self._cos01(e_key, e_idea)      # 近いほど良い
            sim_key_task = self._cos01(e_key, e_task)      # 近いほど“普通”
            b = sim_key_idea * (1.0 - sim_key_task)        # 橋渡し度
            vals.append(b)

        if not vals:
            return 0.0
        vals = sorted(vals, reverse=True)
        k = max(1, min(topk, len(vals)))
        return float(sum(vals[:k]) / k)


# ===== Policy / LoRA =====
@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any

def load_policy_model(model_name: str) -> ModelBundle:
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = _preferred_dtype()
    base = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, dtype=dtype, device_map=None
    )
    base.to("cuda:0")
    if not hasattr(base, "v_head") or base.v_head is None:
        from trl.models.modeling_value_head import ValueHead
        hs = base.pretrained_model.config.hidden_size if hasattr(base, "pretrained_model") else base.config.hidden_size
        base.v_head = ValueHead(hs).to("cuda:0")
    try: base.config.use_cache = True
    except: pass
    return ModelBundle(model=base, tokenizer=tok)

def apply_lora_inplace(bundle: ModelBundle, r: int, alpha: int, dropout: float):
    if not r or r <= 0:
        return
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules
    )
    policy = bundle.model
    base = getattr(policy, "pretrained_model", policy)
    for p in base.parameters():
        p.requires_grad = False
    peft_base = get_peft_model(base, lcfg)
    if hasattr(policy, "pretrained_model"):
        policy.pretrained_model = peft_base; bundle.model = policy
    else:
        bundle.model = peft_base
    bundle.model.to("cuda:0")
    if getattr(bundle.model, "v_head", None) is not None:
        bundle.model.v_head.to("cuda:0")

# ===== BRIDGER salvage (軽量サルベージも残す) =====
_BRIDGER_ALIAS_PATTS = [
    r'^\s*keys?\s*[:=]\s*(.+)$',
    r'^\s*ACTIONABLE\s*KEYS?\s*[:=]\s*(.+)$',
    r'^\s*KEYS?\s*[:=]\s*(.+)$',
]
def _salvage_keys_from_text(text: str) -> List[str]:
    if not text: return []
    s = _desmart(_strip_fences(text))
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    found = []
    for ln in lines:
        for pat in _BRIDGER_ALIAS_PATTS:
            m = re.match(pat, ln, flags=re.IGNORECASE)
            if m:
                payload = m.group(1)
                if not payload.lstrip().startswith(("{","[")):
                    toks = re.split(r'[,\u3001]|["“”\'’]|[\s]+', payload)
                    toks = [t for t in (tok.strip("-_/|. ") for tok in toks) if t]
                    found.extend(toks)
    uniq, seen = [], set()
    for k in found:
        k2 = k[:64]; kl = k2.lower()
        if kl and kl not in seen:
            if len(k2.split()) <= 6:
                uniq.append(k2); seen.add(kl)
    return uniq[:9]

def _enforce_keys_only_json(j) -> Dict[str, Any]:
    keys: List[str] = []
    if isinstance(j, dict):
        cands = []
        for k, v in j.items():
            nk = _desmart(str(k)).strip().strip('"').strip("'").lower()
            if nk in {"keys","key","ideas","bisociation_keys"}:
                cands.append(v)
        for c in cands:
            if isinstance(c, str):
                keys.extend([w.strip() for w in re.split(r"[,\n;]+", c) if w.strip()])
            elif isinstance(c, list):
                keys.extend([str(x).strip() for x in c if str(x).strip()])
    elif isinstance(j, list):
        keys = [str(x).strip() for x in j if str(x).strip()]
    uniq, seen = [], set()
    for k in keys:
        k2 = k[:64]; kl = k2.lower()
        if kl and kl not in seen:
            uniq.append(k2); seen.add(kl)
    return {"keys": uniq[:9]}

def _fill(template: str, mapping: Dict[str, str]) -> str:
    out = template
    for k, v in mapping.items():
        out = out.replace(f"[[{k}]]", v)
    return out

# def run_pipeline_three_agents(bundle: ModelBundle, task: str) -> Dict[str, Any]:
#     # BRIDGER
#     bridger_prompt = _fill(BRIDGER_PROMPT, {"TASK": task})
#     bridger_out = run_generate_cuda(
#         bundle.model, bundle.tokenizer, bridger_prompt,
#         GEN_BRIDGER, stopper=JSONKeysStopper(bundle.tokenizer)
#     )

#     # 未完JSON修復 → JSON抽出 → 直抜き
#     repaired = _repair_minified_json(bridger_out)
#     bj = extract_first_json(repaired) or extract_first_json(bridger_out)

#     def _enforce_keys_only_json_safe(bj_obj) -> Dict[str, Any]:
#         try:
#             return _enforce_keys_only_json(bj_obj)
#         except Exception:
#             return {"keys": []}

#     bridger_json = _enforce_keys_only_json_safe(bj) if bj is not None else {"keys": []}
#     if not bridger_json.get("keys"):
#         hard = _salvage_keys_from_any(bridger_out)
#         if hard: bridger_json["keys"] = hard

#     if not bridger_json.get("keys"):
#         rough = [w for w in re.split(r"[\s,/|]+", str(task)) if len(w) > 3][:6]
#         bridger_json["keys"] = rough or ["LLM","Robotics","Grounding","Multimodal","Evaluation"]
#         print("[WARN] BRIDGER returned no usable keys. raw:\n", (bridger_out or "")[:2000])

#     # PIONEER
#     pioneer_prompt = _fill(PIONEER_PROMPT, {
#         "TASK": task,
#         "KEYS": json.dumps(bridger_json.get("keys", []), ensure_ascii=False)
#     })
#     pioneer_out = run_generate_cuda(bundle.model, bundle.tokenizer, pioneer_prompt, GEN_PIONEER)
#     pj_raw = extract_first_json(pioneer_out)
#     pioneer_json = _normalize_ideas_obj(pj_raw) if pj_raw is not None else {"ideas": [], "rationale": ""}

#     # OBSERVER
#     observer_prompt = _fill(OBSERVER_PROMPT, {
#         "TASK": task,
#         "PIONEER_JSON": json.dumps(pioneer_json, ensure_ascii=False)
#     })
#     observer_out = run_generate_cuda(bundle.model, bundle.tokenizer, observer_prompt, GEN_OBSERVER)
#     oj_raw = extract_first_json(observer_out)
#     observer_json = _normalize_ideas_obj(oj_raw) if oj_raw is not None else {"ideas": [], "rationale": ""}

#     clean_keys = []
#     for k in bridger_json.get("keys", []):
#         k2 = k.strip().lower()
#         # 不要語の除外（任意）
#         if k2 in {"overview","introduction","analysis","method","approach"}:
#             continue
#         clean_keys.append(k.strip())

#     return {
#         "bridger_raw": str(bridger_out),
#         "bridger": {"keys": clean_keys[:9]},
#         "pioneer_raw": str(pioneer_out),
#         "pioneer": pioneer_json,
#         "observer_raw": str(observer_out),
#         "observer": observer_json,
#     }

def run_pipeline_three_agents(bundle: ModelBundle, task: str) -> Dict[str, Any]:
    # ===== BRIDGER（差し替え）=====
    # BRIDGER
    keys, raw_outs = gen_bridger_keys(bundle, task, n=2)
    clean_keys = [k.strip() for k in keys
                  if k and k.strip().lower() not in {"overview","introduction","analysis","method","approach"}]
    bridger_json = {"keys": clean_keys[:9]}
    bridger_raw_joined = ("\n---\n".join(raw_outs))[:2000]

    # ===== PIONEER =====
    pioneer_prompt = _fill(PIONEER_PROMPT, {
        "TASK": task,
        "KEYS": json.dumps(bridger_json["keys"], ensure_ascii=False)
    })
    pioneer_out = run_generate_cuda(bundle.model, bundle.tokenizer, pioneer_prompt, GEN_PIONEER)
    pj = extract_first_json(pioneer_out) or {}
    ideas = pj.get("ideas") or []
    if isinstance(ideas, dict): ideas = [ideas]
    if not isinstance(ideas, list): ideas = []
    rationale = pj.get("rationale") or ""
    pj_raw = extract_first_json(pioneer_out) or {}
    pioneer_json = _normalize_ideas_obj(pj_raw)

    # ===== OBSERVER =====
    observer_prompt = _fill(OBSERVER_PROMPT, {
        "TASK": task,
        "PIONEER_JSON": json.dumps(pioneer_json, ensure_ascii=False)
    })
    observer_out = run_generate_cuda(bundle.model, bundle.tokenizer, observer_prompt, GEN_OBSERVER)
    oj = extract_first_json(observer_out) or {}
    o_ideas = oj.get("ideas") or []
    if isinstance(o_ideas, dict): o_ideas = [o_ideas]
    if not isinstance(o_ideas, list): o_ideas = []
    o_rat = oj.get("rationale") or ""

    oj_raw = extract_first_json(observer_out) or {}
    observer_json = _normalize_ideas_obj(oj_raw)

    return {
        "bridger_raw": bridger_raw_joined,   # BRIDGERはマージなので raw は空でもOK（必要なら連結して入れても良い）
        "bridger": {"keys": bridger_json["keys"]},
        "pioneer_raw": str(pioneer_out),
        "pioneer": pioneer_json,
        "observer_raw": str(observer_out),
        "observer": observer_json,
    }

# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", type=str, required=True)

    # IRM
    ap.add_argument("--irm-model-dir", type=str, required=True)
    ap.add_argument("--irm-calib-path", type=str, default=None)
    ap.add_argument("--irm-max-len", type=int, default=512)
    ap.add_argument("--irm-device", type=str, default="cuda:1")

    # Seeds
    ap.add_argument("--seeds-path", type=str, required=True)
    ap.add_argument("--max-seeds", type=int, default=None)

    # PPO / Train
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--mini-batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=5e-6)

    # Generation
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)

    # LoRA
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    # Reward scaling / calibration
    ap.add_argument("--reward-mode", type=str, default="calib",
                    choices=["prob","logit","ztemp","platt","calib"])
    ap.add_argument("--reward-temp", type=float, default=4.0)  # ztemp 温度
    ap.add_argument("--log-every-n", type=int, default=20)
    ap.add_argument("--policy-device", type=str, default="cuda:0")

    # Optional eval hook (off by default)
    ap.add_argument("--eval-every", type=int, default=0)

    # Bridge reward
    ap.add_argument("--bridge-weight", type=float, default=0.0,
                    help="Weight for bridge reward (0=off). Typical 0.1~0.3")
    ap.add_argument("--bridge-emb-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                    help="Encoder for bridge reward (text embedding)")
    ap.add_argument("--bridge-topk", type=int, default=3, help="Top-K averaging for bridge score")
    ap.add_argument("--bridge-device", type=str, default=None,
                    help="Device for bridge encoder; default uses irm-device")

    args = ap.parse_args()

    # CUDA sanity
    assert_cuda_device("cuda:0")
    assert_cuda_device("cuda:1")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_default_dtype_bf16_if_available()

    # Policy
    bundle = load_policy_model(args.model_name)
    apply_lora_inplace(bundle, args.lora_r, args.lora_alpha, args.lora_dropout)
    print("[SANITY] policy device ->", next(bundle.model.parameters()).device)
    if str(next(bundle.model.parameters()).device) != "cuda:0":
        raise RuntimeError("Policy model is not on cuda:0.")
    if getattr(bundle.model, "v_head", None) is not None:
        bundle.model.v_head.to("cuda:0")

    # IRM
    irm = IRMScorer(args.irm_model_dir, max_len=args.irm_max_len, device=args.irm_device)
    irm.load_calib(args.irm_calib_path)

    # Bridge (optional)
    bridge = None
    if args.bridge_weight > 0.0:
        bdev = args.bridge_device if args.bridge_device else args.irm_device
        try:
            bridge = BridgeScorer(args.bridge_emb_model, device=bdev)
            print(f"[INIT] BridgeScorer loaded on {bdev} with {args.bridge_emb_model}")
        except Exception as e:
            print(f"[WARN] BridgeScorer disabled (load failed): {e}")
            bridge = None

    # Prompts
    prompts = load_prompts_from_jsonl(args.seeds_path, args.max_seeds)
    if not prompts:
        raise RuntimeError("Empty seeds. Check --seeds-path.")

    # PPO config
    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        seed=42,
        init_kl_coef=0.02,
        target_kl=0.05,
        use_score_scaling=False,
        use_score_norm=False,
    )
    if hasattr(ppo_cfg, "use_reference_model"):
        setattr(ppo_cfg, "use_reference_model", False)

    dataset = PromptDataset(prompts)
    class DummyCollator:
        def __call__(self, batch): return batch

    trainer = PPOTrainer(
        config=ppo_cfg,
        model=bundle.model,
        ref_model=bundle.model,
        tokenizer=bundle.tokenizer,
        dataset=dataset,
        data_collator=DummyCollator(),
    )
    trainer.model.to("cuda:0").eval()
    if getattr(trainer, "ref_model", None) is not None:
        trainer.ref_model.to("cuda:0").eval()

    # Generation cfg
    gen_cfg = GenCfg(
        max_new_tokens=args.max_new_tokens,
        temperature=max(0.0, min(args.temperature, 1.0)),
        top_p=args.top_p,
        top_k=args.top_k,
    )

    # W&B
    if _HAS_WANDB:
        wandb.init(project="CREA-Bridge", config=vars(args))
        samples_table = wandb.Table(columns=["step","example_task","bridger_raw","pioneer_raw","observer_raw"])
    else:
        samples_table = None

    print("[INFO] PPO start: policy=cuda:0, IRM=cuda:1, response=Observer JSON")
    total_updates = args.total_steps
    pbar = trange(total_updates, desc="PPO Updates", unit="upd", dynamic_ncols=True) if _HAS_TQDM else None
    ema = EMA(0.9)
    next_idx = 0
    skipped_cnt_total = 0

    for upd in range(total_updates):
        t0 = time.time()
        batch_query_tensors: List[torch.Tensor] = []
        batch_response_tensors: List[torch.Tensor] = []
        batch_rewards: List[torch.Tensor] = []
        batch_logits_list: List[torch.Tensor] = []
        r_irm_list, r_bridge_list, r_total_list = [], [], []

        eff_bsz = 0
        sample_rows_this_step = []  # ← 追加：このステップの可視化行を蓄積
        last_out_dict = {}
        last_task_str = ""

        for b in range(args.batch_size):
            task = prompts[next_idx % len(prompts)]
            next_idx += 1
            try:
                out = run_pipeline_three_agents(bundle, task)
            except Exception as e_pipeline:
                print(f"[DEBUG] pipeline failed at upd {upd+1} (b={b}): {repr(e_pipeline)}")
                skipped_cnt_total += 1
                continue  # ← 重要：壊れたサンプルは完全スキップ

            # keys チェック
            keys_now = out.get("bridger", {}).get("keys", [])
            if not isinstance(keys_now, list):
                keys_now = []
            print(f"[DBG] step={upd+1} b={b} keys_count={len(keys_now)}")
            if len(keys_now) < 3:
                skipped_cnt_total += 1
                continue

            # アイデア選択
            obs_ideas = out.get("observer", {}).get("ideas", [])
            if obs_ideas:
                choice = obs_ideas[0] if isinstance(obs_ideas, list) else {}
                title = (choice.get("title") or "Untitled")[:200] if isinstance(choice, dict) else "Untitled"
                abstract = (choice.get("abstract") or "")[:4000] if isinstance(choice, dict) else ""
                trainable_text = f"{title}\n{abstract}"
            else:
                raw = out.get("observer_raw") or out.get("pioneer_raw") or out.get("bridger_raw") or ""
                lines = [ln.strip() for ln in str(raw).splitlines() if ln.strip()]
                title = (lines[0] if lines else "Untitled")[:120]
                abstract = str(raw)[:1000]
                trainable_text = f"{title}\n{abstract}"

            # 生成（policy.generate）
            try:
                enc = bundle.tokenizer(
                    trainable_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max(128, gen_cfg.max_new_tokens),
                )
                enc = {k: v.to("cuda:0") for k, v in enc.items()}
                query_len = enc["input_ids"].shape[1]

                with torch.no_grad():
                    gen_out = bundle.model.generate(
                        **enc,
                        do_sample=(gen_cfg.temperature > 0),
                        temperature=gen_cfg.temperature,
                        top_p=gen_cfg.top_p,
                        top_k=gen_cfg.top_k,
                        max_new_tokens=gen_cfg.max_new_tokens,
                        pad_token_id=bundle.tokenizer.eos_token_id,
                        eos_token_id=bundle.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                resp_ids = gen_out[0, query_len:]
                if resp_ids.numel() == 0:
                    resp_ids = gen_out[0, -1:].detach()

                # clip for PPO stability
                q_ids = enc["input_ids"][0]
                if q_ids.shape[0] > MAX_QUERY_TOKENS_FOR_PPO:
                    q_ids = q_ids[-MAX_QUERY_TOKENS_FOR_PPO:]
                if resp_ids.shape[0] > MAX_RESP_TOKENS_FOR_PPO:
                    resp_ids = resp_ids[:MAX_RESP_TOKENS_FOR_PPO]

                batch_query_tensors.append(q_ids)
                batch_response_tensors.append(resp_ids)

                # ---- Rewards ----
                r_irm = float(irm.score_text(title, abstract, reward_mode=args.reward_mode, reward_temp=args.reward_temp))
                r_bridge = 0.0
                if bridge is not None:
                    task_text_for_bridge = str(task)
                    idea_text_for_bridge = f"{title}\n{abstract}"
                    keys_for_bridge = [k for k in (keys_now or []) if isinstance(k, str)]
                    try:
                        r_bridge = float(bridge.score(task_text_for_bridge, keys_for_bridge, idea_text_for_bridge, topk=int(args.bridge_topk)))
                    except Exception as e_b:
                        print(f"[DEBUG] bridge score failed: {e_b}")
                        r_bridge = 0.0

                r_total = max(0.0, min(1.0, r_irm)) + float(args.bridge_weight) * max(0.0, min(1.0, r_bridge))
                rew_t = torch.tensor(r_total, dtype=torch.float32, device="cuda:0")
                batch_rewards.append(rew_t)

                # IRM logits（debug）
                lg = irm.logits_text(title, abstract)  # cuda:1
                batch_logits_list.append(lg.detach().float().cpu().view(-1))

                r_irm_list.append(r_irm)
                r_bridge_list.append(r_bridge)
                r_total_list.append(r_total)

                eff_bsz += 1

                last_out_dict = out
                last_task_str = str(task)

                # W&B 行バッファに積む
                if _HAS_WANDB:
                    lod = out if isinstance(out, dict) else {}
                    sample_rows_this_step.append([
                        upd + 1,
                        str(task)[:2000],
                        safe_get(lod, "bridger_raw", "")[:2000],
                        safe_get(lod, "pioneer_raw", "")[:2000],
                        safe_get(lod, "observer_raw", "")[:2000],
                    ])

            except Exception as e_one:
                print(f"[DEBUG] sample build failed at upd {upd+1} (b={b}): {repr(e_one)}")
                continue

        # バッチに有効サンプルが無ければこのステップはスキップ
        if eff_bsz == 0:
            print(f"[INFO] skip whole update {upd+1}: no valid samples (skipped so far={skipped_cnt_total})")
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"epoch": f"{args.ppo_epochs}/{args.ppo_epochs}", "reward":"NaN","bsz":0})
            # （必要ならここでダミー行を samples_table に入れても良い）
            continue

        # 動的バッチ調整
        if getattr(trainer, "config", None) is not None:
            trainer.config.batch_size = eff_bsz
            if trainer.config.mini_batch_size > eff_bsz:
                trainer.config.mini_batch_size = eff_bsz

        # PPO step
        try:
            print(f"[DBG] effective_batch_size={eff_bsz}, ppo(batch={trainer.config.batch_size}, mini={trainer.config.mini_batch_size})")
            trainer.step(batch_query_tensors, batch_response_tensors, batch_rewards)

            vals = [float(r.item()) for r in batch_rewards]
            p50 = float(np.median(vals))
            p90 = float(np.percentile(vals, 90))
            var = float(np.var(vals))
            with torch.no_grad():
                batch_mean = torch.stack(batch_rewards).mean().item()
            rew_for_log = float(batch_mean)
        except Exception as e_step:
            print(f"[DEBUG] PPO step failed at upd {upd+1}: {repr(e_step)}")
            import traceback; traceback.print_exc()
            rew_for_log = float("nan"); p50 = p90 = var = float("nan")

        # W&B logs
        step_time = time.time() - t0
        ema_val = ema.update(rew_for_log if (rew_for_log == rew_for_log) else float("nan"))
        rows_added = len(sample_rows_this_step)
        print(f"[DBG] step={upd+1} add_rows={rows_added} eff_bsz={eff_bsz}")

        if _HAS_WANDB:
            log_dict = {
                "step": upd + 1,
                "reward/batch_mean": float(rew_for_log) if (rew_for_log == rew_for_log) else float("nan"),
                "reward/p50_median": p50,
                "reward/p90_top": p90,
                "reward/variance": var,
                "reward/ema": float(ema_val) if (ema_val == ema_val) else float("nan"),
                "time/step_sec": step_time,
                "effective_batch_size": eff_bsz,
                "data/skip_total": skipped_cnt_total,
            }
            if r_irm_list:
                log_dict.update({
                    "reward/irm": float(np.mean(r_irm_list)),
                    "reward/bridge": float(np.mean(r_bridge_list)),
                    "reward/total": float(np.mean(r_total_list)),
                })
            wandb.log(log_dict, step=upd + 1, commit=False)

            # logits histogram
            try:
                if len(batch_logits_list) > 0:
                    arr = torch.stack(batch_logits_list)  # [B,C] or [B,1]
                    if arr.ndim == 2 and arr.shape[1] > 1:
                        max_logit = arr.max(dim=1).values.numpy()
                        wandb.log({
                            "debug/irm/max_logit_mean": float(np.mean(max_logit)),
                            "debug/irm/max_logit_std": float(np.std(max_logit)),
                            "debug/irm/max_logit_hist": wandb.Histogram(max_logit),
                        }, step=upd + 1, commit=False)
                    else:
                        v = arr.view(-1).numpy()
                        wandb.log({
                            "debug/irm/logit_mean": float(np.mean(v)),
                            "debug/irm/logit_std": float(np.std(v)),
                            "debug/irm/logit_hist": wandb.Histogram(v),
                        }, step=upd + 1, commit=False)
            except Exception as e:
                print(f"[DEBUG] wandb logits log skipped at upd {upd+1}: {e}")

            # samples/outputs: そのステップで貯めた行を追加してからテーブルをログ
            if (upd + 1) % max(1, args.log_every_n) == 0:
                try:
                    if samples_table is not None and rows_added > 0:
                        for row in sample_rows_this_step:
                            samples_table.add_data(*row)
                    wandb.log({"samples/outputs": samples_table}, step=upd + 1, commit=True)
                except Exception as e:
                    print(f"[DEBUG] wandb table log skipped at upd {upd+1}: {e}")

        if pbar:
            pbar.update(1)
            pbar.set_postfix({
                "epoch": f"{args.ppo_epochs}/{args.ppo_epochs}",
                "reward": f"{rew_for_log:.3f}" if (rew_for_log == rew_for_log) else "NaN",
                "bsz": eff_bsz
            })

    if pbar:
        pbar.close()
    print("[INFO] PPO training done.")

if __name__ == "__main__":
    try:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if vis is not None:
            if len([x for x in vis.split(",") if x.strip() != ""]) < 2:
                raise RuntimeError("Set CUDA_VISIBLE_DEVICES with at least two GPUs, e.g., '0,1'.")
        if torch.cuda.device_count() < 2:
            raise RuntimeError("Fewer than 2 GPUs visible. Check nvidia-smi & CUDA_VISIBLE_DEVICES.")
        print("[ATTN] attention implementation = sdpa")
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}")
        sys.exit(1)
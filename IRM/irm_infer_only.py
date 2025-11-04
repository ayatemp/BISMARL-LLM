# irm_infer_only.py  —— 推論専用（Trainer/peft/accelerate 依存を排除）
import os, json, math
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def _apply_percentile_map(x: float, sorted_preds: List[float]) -> float:
    if not sorted_preds:
        return float(np.clip((x - 1.0)/9.0, 0.0, 1.0))
    import bisect as _bisect
    idx = _bisect.bisect_left(sorted_preds, x)
    n = len(sorted_preds)
    if n <= 1: return 0.5
    return float(np.clip(idx / (n - 1), 0.0, 1.0))

class IRMScorer:
    """
    - raw_score: 回帰モデル出力（uncertainty学習のモデルなら μ を返す想定）
    - reward: Isotonic(あれば) or 分位キャリブで [0,1]
    - スライディングウィンドウ推論対応
    """
    def __init__(self, model_dir: str, max_length: int = 512,
                 window_stride_ratio: float = 0.5, agg: str = "median", device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
        self.max_length = max_length
        self.stride = int(max_length * window_stride_ratio)
        self.agg = agg  # "mean" | "median" | "wmean"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 校正読み込み
        self.calib = None
        self.iso = None
        iso_path = os.path.join(model_dir, "reward_iso.joblib")
        calib_path = os.path.join(model_dir, "reward_calibration.json")
        if os.path.exists(iso_path):
            try:
                # scikit-learn が無ければ失敗 → None
                import joblib
                self.iso = joblib.load(iso_path)
            except Exception:
                self.iso = None
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r") as f:
                    obj = json.load(f)
                self.calib = obj.get("sorted_preds", None)
            except Exception:
                self.calib = None

    def _encode_windows(self, text: str):
        ids = self.tokenizer(text, truncation=False, padding=False, return_tensors=None)["input_ids"]
        chunks = []
        start = 0
        while start < len(ids):
            end = start + self.max_length
            ch = ids[start:end]
            if not ch: break
            chunks.append(ch)
            if end >= len(ids): break
            start = max(0, end - self.stride)
            if start == 0 and end >= len(ids): break
        return chunks

    @torch.no_grad()
    def _score_single(self, text: str) -> Tuple[float, Optional[float]]:
        # logits が [B], [B,1], [B,2(mu,logvar)] のどれでも動くように実装
        def _extract_mu_var(logits_tensor):
            if logits_tensor.dim() == 1:
                mu = logits_tensor
                logvar = None
            elif logits_tensor.size(-1) == 1:
                mu = logits_tensor.squeeze(-1)
                logvar = None
            else:
                mu = logits_tensor[..., 0]
                logvar = logits_tensor[..., 1].clamp(min=-10, max=10)
            return mu, logvar

        chunks = self._encode_windows(text)
        preds, lens, logvars = [], [], []

        for ch in chunks:
            batch = self.tokenizer.prepare_for_model(
                ch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in batch and batch[k].dim() == 1:
                    batch[k] = batch[k].unsqueeze(0)
            lens.append(len(ch))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            mu, logvar = _extract_mu_var(out.logits.detach().cpu())
            preds.append(mu.squeeze(0).item())
            if logvar is not None:
                logvars.append(logvar.squeeze(0).item())

        if not preds:
            batch = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in batch and batch[k].dim() == 1:
                    batch[k] = batch[k].unsqueeze(0)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            mu, logvar = _extract_mu_var(out.logits.detach().cpu())
            mu_val = mu.squeeze(0).item()
            unc = float(math.exp(0.5 * logvar.squeeze(0).item())) if logvar is not None else None
            return mu_val, unc

        if self.agg == "mean":
            mu_val = float(np.mean(preds))
            unc = float(np.mean([math.exp(0.5 * lv) for lv in logvars])) if logvars else None
        elif self.agg == "wmean":
            w = np.array(lens, dtype=float); w = w / (w.sum() + 1e-8)
            mu_val = float((w * np.array(preds)).sum())
            unc = float((w * np.array([math.exp(0.5 * lv) for lv in logvars])).sum()) if logvars else None
        else:
            mu_val = float(np.median(preds))
            unc = float(np.median([math.exp(0.5 * lv) for lv in logvars])) if logvars else None

        return mu_val, unc

    def _to_reward(self, raw: float, uncertainty: Optional[float]) -> float:
        if self.iso is not None:
            try:
                r = float(self.iso.predict([raw])[0])
            except Exception:
                r = None
            if r is not None:
                if uncertainty is not None:
                    r = r * float(1.0 / (1.0 + 0.1 * uncertainty))
                return float(np.clip(r, 0.0, 1.0))
        if self.calib:
            r = _apply_percentile_map(raw, self.calib)
            if uncertainty is not None:
                r = r * float(1.0 / (1.0 + 0.1 * uncertainty))
            return float(np.clip(r, 0.0, 1.0))
        r = float(np.clip((raw - 1.0)/9.0, 0.0, 1.0))
        if uncertainty is not None:
            r = r * float(1.0 / (1.0 + 0.1 * uncertainty))
        return float(np.clip(r, 0.0, 1.0))

    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        outs = []
        for tx in texts:
            raw, unc = self._score_single(tx)
            outs.append({"raw_score": raw, "reward": self._to_reward(raw, unc), "uncertainty": (float(unc) if unc is not None else None)})
        return outs

    def score_ideas(self, titles: List[str], bodies: List[str]) -> List[Dict[str, float]]:
        texts = [(t or "").strip() + "\n\n" + (b or "").strip() for t, b in zip(titles, bodies)]
        return self.score_texts(texts)
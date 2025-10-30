# irm_iclr.py
# coding: utf-8
"""
改良版 IRM（Idea Reward Model）
- 入力: ICLRデータ（.parquet / .json / .jsonl）
- 目的: タイトル+アブスト → スコア回帰（順序性/年差/ノイズを考慮した学習・報酬化）
- 主な改善:
  1) loss_type: mse / mae / huber / rank（順位補助損失）
  2) target_type: raw / year_z（年別z正規化; 学習分割の統計を保存＆評価時に再利用）
  3) 推論時スライディングウィンドウ（集約: mean/median/wmean、stride可変）
  4) SciBERTをデフォルトモデル
  5) 検証予測の分位キャリブで0–1報酬安定化（reward_calibration.json）
  6) 分位ビニングの層化分割


実行例
python irm_iclr.py train \
  --data_path data/iclr/ \
  --out ./irm_sci_huber_z_splitstats \
  --model_name allenai/scibert_scivocab_uncased \
  --epochs 3 --bs 16 --lr 2e-5 --max_len 512 \
  --loss_type huber --target_type year_z --accept_threshold 6.0 --add_year_tag
"""

import os, re, json, glob, math, random, bisect, inspect
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import pyarrow.parquet as pq

RATING_PAT = re.compile(r"^\s*(\d+(\.\d+)?)")

def extract_numeric_rating(text: str) -> Optional[float]:
    if text is None: return None
    m = RATING_PAT.match(str(text))
    if not m: return None
    try:
        return float(m.group(1))
    except Exception:
        return None

# -----------------------------
# ローダ（Parquet/JSON/JSONL）
# -----------------------------
def load_iclr_parquet(parquet_path: str) -> List[Dict]:
    pf = pq.ParquetFile(parquet_path)
    cols = pf.schema_arrow.names

    title_col = "title"
    abs_candidates = ["abstract", "abstract_text", "paper_abstract"]
    score_candidates = ["scores", "score", "ratings", "review_scores", "scores_mean", "avg_score"]
    year_candidates = ["year", "conference_year"]
    decision_candidates = ["decision", "final_decision", "accept"]

    abs_col = next((c for c in abs_candidates if c in cols), None)
    score_col = next((c for c in score_candidates if c in cols), None)
    year_col = next((c for c in year_candidates if c in cols), None)
    dec_col = next((c for c in decision_candidates if c in cols), None)

    need_cols = [c for c in [title_col, abs_col, score_col, year_col, dec_col] if c and c in cols]
    table = pf.read(columns=need_cols, use_pandas_metadata=False)

    def col_list(name: str) -> List:
        if not name or name not in need_cols:
            return [None for _ in range(table.num_rows)]
        return table.column(name).to_pylist()

    titles  = col_list(title_col)
    abstracts = col_list(abs_col)
    scores_raw = col_list(score_col)
    years   = col_list(year_col)
    decis   = col_list(dec_col)

    recs: List[Dict] = []
    for t, a, sv, y, d in zip(titles, abstracts, scores_raw, years, decis):
        title = str(t or "").strip()
        abstract = str(a or "").strip()
        score = None
        try:
            if isinstance(sv, (list, tuple, np.ndarray)):
                arr = [float(x) for x in sv if x is not None and not (isinstance(x, float) and np.isnan(x))]
                if arr: score = float(np.mean(arr))
            elif isinstance(sv, str):
                num = extract_numeric_rating(sv)
                if num is not None: score = float(num)
            elif sv is not None and not (isinstance(sv, float) and np.isnan(sv)):
                score = float(sv)
        except Exception:
            pass

        year = None
        try:
            if y is not None:
                ystr = str(y).strip()
                ys = re.findall(r"\d{4}", ystr)
                year = int(ys[0]) if ys else None
        except Exception:
            year = None

        decision = None
        if isinstance(d, str):
            decision = d.lower()
        elif isinstance(d, (int, float)):
            decision = "accept" if d else "reject"

        recs.append({"title": title, "abstract": abstract, "score": score, "year": year, "decision": decision})
    return recs

def load_iclr_records_from_dir(data_dir: str) -> List[Dict]:
    paths = []
    paths += glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)

    records: List[Dict] = []
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        records.append(obj)
                    except Exception:
                        pass
        else:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    records.extend(obj)
                else:
                    records.append(obj)
            except Exception:
                pass
    return records

def _records_from_path(data_path: str) -> List[Dict]:
    if os.path.isdir(data_path):
        pq_files = glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True)
        recs = []
        if pq_files:
            for p in sorted(pq_files):
                recs.extend(load_iclr_parquet(p))
        else:
            recs = load_iclr_records_from_dir(data_path)
    else:
        if data_path.endswith(".parquet"):
            recs = load_iclr_parquet(data_path)
        elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
            recs = []
            if data_path.endswith(".jsonl"):
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try: recs.append(json.loads(line))
                            except Exception: pass
            else:
                with open(data_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list): recs.extend(obj)
                else: recs.append(obj)
        else:
            raise ValueError("data_path はディレクトリまたは .parquet/.json/.jsonl を指定してください。")
    return recs

# -----------------------------
# データセット化
# -----------------------------
@dataclass
class DataConfig:
    data_path: str
    model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 512
    seed: int = 123
    train_ratio: float = 0.9
    min_len: int = 30
    target_type: str = "raw"   # "raw" | "year_z"
    accept_threshold: float = 6.0
    add_year_tag: bool = True  # テキスト先頭に [YEAR=2024]

def make_examples(cfg: DataConfig) -> List[Dict]:
    recs = _records_from_path(cfg.data_path)
    examples: List[Dict] = []
    for rec in recs:
        title = (rec.get("title") or "").strip()
        abs_  = (rec.get("abstract") or rec.get("abstract_text") or "").strip()
        text0 = (title + "\n\n" + abs_).strip()
        if len(text0) < cfg.min_len: continue

        score = rec.get("score", None)
        if score is None:
            reviews = rec.get("reviews") or rec.get("meta_reviews") or []
            if isinstance(reviews, dict):
                reviews = [reviews]
            ratings = []
            for rv in reviews:
                cand = rv.get("rating") or rv.get("final_decision") or rv.get("recommendation")
                r = extract_numeric_rating(str(cand)) if cand is not None else None
                if r is not None and 0.5 <= r <= 10.5:
                    ratings.append(r)
            if ratings:
                score = float(np.mean(ratings))
        if score is None: continue

        year = rec.get("year", None)
        text = f"[YEAR={year}]\n{text0}" if (cfg.add_year_tag and year) else text0
        accept = 1 if float(score) >= cfg.accept_threshold else 0
        examples.append({"text": text, "score": float(score), "year": year, "accept": accept})
    return examples

def stratified_split_by_score_bins(examples: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    scores = np.array([ex["score"] for ex in examples], dtype=float)
    qs = np.quantile(scores, np.linspace(0, 1, 11))
    def bin_id(s):
        idx = int(np.clip(np.searchsorted(qs, s, side="right") - 1, 0, 9))
        return idx
    bins = [bin_id(s) for s in scores]
    train, valid = [], []
    for b in range(10):
        idxs = [i for i, bb in enumerate(bins) if bb == b]
        rng.shuffle(idxs)
        ntr = int(len(idxs) * train_ratio)
        for i in idxs[:ntr]: train.append(examples[i])
        for i in idxs[ntr:]: valid.append(examples[i])
    return train, valid

def compute_year_stats(examples: List[Dict]) -> Tuple[Dict[Optional[int], Tuple[float,float]], Tuple[float,float]]:
    df = pd.DataFrame(examples)
    df["year_"] = df["year"].fillna(-1)
    g = df.groupby("year_")["score"]
    stats = {}
    for k, s in g:
        mu = float(s.mean())
        sd = float(s.std(ddof=0))
        stats[(None if k == -1 else int(k))] = (mu, sd)
    mu_all = float(df["score"].mean())
    sd_all = float(df["score"].std(ddof=0))
    sd_all = sd_all if sd_all > 1e-8 else 1.0
    return stats, (mu_all, sd_all)

def z_transform_with_stats(examples: List[Dict],
                           stats: Dict[Optional[int], Tuple[float,float]],
                           default_stats: Tuple[float,float]) -> List[Dict]:
    out = []
    for ex in examples:
        y = ex.get("year", None)
        mu, sd = stats.get(y, default_stats)
        sd = sd if sd > 1e-8 else 1e-8
        ex2 = dict(ex)
        ex2["target"] = float((ex["score"] - mu) / sd)
        out.append(ex2)
    return out

def _safe_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    sig_params = set(sig.parameters.keys())
    valid = {k: v for k, v in kwargs.items() if k in sig_params}
    if "evaluation_strategy" not in sig_params and "evaluate_during_training" in sig_params:
        valid["evaluate_during_training"] = True
    if "load_best_model_at_end" in sig_params:
        ev = valid.get("evaluation_strategy", "no")
        if ev == "no" or "save_strategy" not in sig_params:
            valid["load_best_model_at_end"] = False
            valid.pop("metric_for_best_model", None)
            valid.pop("greater_is_better", None)
    return TrainingArguments(**valid)

def make_dataset(cfg: DataConfig):
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    examples = make_examples(cfg)
    if len(examples) < 100:
        raise RuntimeError(f"学習サンプルが少なすぎます（{len(examples)}件）。データ/列名を確認してください。")

    # 年トークン候補（語彙追加用）
    year_tags = sorted({f"[YEAR={int(ex['year'])}]" for ex in examples if ex.get("year")})

    train_ex, valid_ex = stratified_split_by_score_bins(examples, cfg.train_ratio, cfg.seed)

    # 学習分割の統計で z を作り、valid も学習統計で変換
    year_stats = None
    default_stats = None
    if cfg.target_type == "year_z":
        year_stats, default_stats = compute_year_stats(train_ex)
        train_ex = z_transform_with_stats(train_ex, year_stats, default_stats)
        valid_ex = z_transform_with_stats(valid_ex, year_stats, default_stats)
    else:
        for ex in train_ex: ex["target"] = float(ex["score"])
        for ex in valid_ex: ex["target"] = float(ex["score"])

    ds_train = Dataset.from_list(train_ex)
    ds_valid = Dataset.from_list(valid_ex)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    # 年トークンを語彙に追加（学習可能に）
    if cfg.add_year_tag and year_tags:
        tokenizer.add_tokens(year_tags, special_tokens=False)

    def tok(batch):
        out = tokenizer(
            batch["text"],
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
        )
        out["labels"] = [float(x) for x in batch["target"]]
        out["score"]  = [float(x) for x in batch["score"]]
        out["accept"] = [int(x) for x in batch["accept"]]
        out["year"]   = [int(y) if (y is not None) else -1 for y in batch["year"]]
        return out

    dsd = DatasetDict({"train": ds_train, "validation": ds_valid})
    dsd = dsd.map(tok, batched=True, remove_columns=["text", "target", "score", "accept", "year"])
    dsd.set_format(type="torch")

    meta = {
        "year_stats": year_stats,
        "default_stats": default_stats,
        "year_tags": year_tags,
    }
    return dsd, tokenizer, meta

# -----------------------------
# 学習
# -----------------------------
@dataclass
class TrainConfig:
    output_dir: str = "./irm_iclr_model"
    model_name: str = "allenai/scibert_scivocab_uncased"
    max_length: int = 512
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 123
    warmup_ratio: float = 0.06
    report_to: str = "none"
    fp16: bool = True
    loss_type: str = "huber"     # "mse" | "mae" | "huber" | "rank"
    rank_lambda: float = 0.2     # 順位補助損失の係数
    target_type: str = "year_z"  # "raw" | "year_z"
    accept_threshold: float = 6.0
    add_year_tag: bool = True

def _spearman(preds: np.ndarray, labels: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(preds, labels).correlation)
    except Exception:
        return 0.0

class RankAuxTrainer(Trainer):
    def __init__(self, loss_type="huber", rank_lambda=0.2, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.rank_lambda = rank_lambda
        if loss_type == "huber":
            self._reg_loss = torch.nn.SmoothL1Loss()
        elif loss_type == "mae":
            self._reg_loss = torch.nn.L1Loss()
        else:
            self._reg_loss = torch.nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items()})
        logits = outputs.logits.squeeze(-1)

        if self.loss_type == "huber":
            reg = self._reg_loss(logits, labels)
            loss = reg
        elif self.loss_type == "mae":
            reg = self._reg_loss(logits, labels)
            loss = reg
        elif self.loss_type == "mse":
            reg = self._reg_loss(logits, labels)
            loss = reg
        elif self.loss_type == "rank":
            reg = torch.nn.functional.mse_loss(logits, labels)
            with torch.no_grad():
                idx = torch.randperm(logits.shape[0], device=logits.device)
            pairs = list(zip(idx[:-1], idx[1:]))[:16]
            rank_loss = 0.0
            for i, j in pairs:
                s = torch.sign(labels[i] - labels[j])
                if s == 0:
                    continue
                diff = logits[i] - logits[j]
                rank_loss = rank_loss + torch.log1p(torch.exp(-s * diff))
            if len(pairs) > 0:
                rank_loss = rank_loss / len(pairs)
            else:
                rank_loss = torch.tensor(0.0, device=logits.device)
            loss = reg + self.rank_lambda * rank_loss
        else:
            loss = torch.nn.functional.mse_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

def train_irm(data_path: str, tcfg: TrainConfig):
    torch.manual_seed(tcfg.seed); np.random.seed(tcfg.seed); random.seed(tcfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dcfg = DataConfig(
        data_path=data_path,
        model_name=tcfg.model_name,
        max_length=tcfg.max_length,
        seed=tcfg.seed,
        target_type=tcfg.target_type,
        accept_threshold=tcfg.accept_threshold,
        add_year_tag=tcfg.add_year_tag,
    )
    dsd, tokenizer, meta = make_dataset(dcfg)

    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg.model_name,
        num_labels=1,
        problem_type="regression",
        use_safetensors=True,
    )
    # 追加トークンに合わせて埋め込みを拡張
    model.resize_token_embeddings(len(tokenizer))

    bf16_flag = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    args = _safe_training_args(
        output_dir=tcfg.output_dir,
        learning_rate=tcfg.lr,
        per_device_train_batch_size=tcfg.batch_size,
        per_device_eval_batch_size=tcfg.batch_size,
        num_train_epochs=tcfg.epochs,
        weight_decay=tcfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        seed=tcfg.seed,
        fp16=(tcfg.fp16 and not bf16_flag),
        bf16=(bf16_flag if "bf16" in inspect.signature(TrainingArguments.__init__).parameters else False),
        warmup_ratio=tcfg.warmup_ratio,
        logging_steps=50,
        report_to=tcfg.report_to if tcfg.report_to in ("none", "wandb") else "none",
        save_total_limit=2,
        gradient_accumulation_steps=1,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        preds = preds.reshape(-1).astype(float)
        labels = labels.reshape(-1).astype(float)
        rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
        mae  = float(np.mean(np.abs(preds - labels)))
        spr  = _spearman(preds, labels)
        return {"rmse": rmse, "mae": mae, "spearman": spr}

    trainer = RankAuxTrainer(
        loss_type=tcfg.loss_type,
        rank_lambda=tcfg.rank_lambda,
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,  # FutureWarning対策: processing_class でも可
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(tcfg.output_dir)
    tokenizer.save_pretrained(tcfg.output_dir)

    # 学習時の year_z 統計を保存（評価時に必ず使用）
    if meta.get("year_stats") is not None:
        ys = meta["year_stats"]
        ds = meta["default_stats"]
        stats_obj = {
            "year_stats": { ("" if k is None else str(k)): {"mean": float(v[0]), "std": float(v[1])} for k, v in ys.items() },
            "default": {"mean": float(ds[0]), "std": float(ds[1])},
            "target_type": tcfg.target_type,
        }
        with open(os.path.join(tcfg.output_dir, "year_z_stats.json"), "w") as f:
            json.dump(stats_obj, f)

    # 分位キャリブレーション保存
    eval_out = trainer.predict(dsd["validation"])
    val_preds = eval_out.predictions.reshape(-1).astype(float)
    calib_sorted = np.sort(val_preds).tolist()
    with open(os.path.join(tcfg.output_dir, "reward_calibration.json"), "w") as f:
        json.dump({"sorted_preds": calib_sorted}, f)
    print("[INFO] Saved calibration and year_z_stats (if any).")

# -----------------------------
# 推論（スコアリング + スライディングウィンドウ）
# -----------------------------
def _build_percentile_map(preds: np.ndarray) -> List[float]:
    return np.sort(preds.astype(float)).tolist()

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
    - raw_score: 回帰モデル出力
    - reward: 分位キャリブレーションにもとづく [0,1]
    - 推論時のみスライディングウィンドウ（stride率・集約方式を選択）
    """
    def __init__(self, model_dir: str, max_length: int = 512,
                 window_stride_ratio: float = 0.5, agg: str = "median"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
        self.max_length = max_length
        self.stride = int(max_length * window_stride_ratio)
        self.agg = agg  # "mean" | "median" | "wmean"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.calib = None
        calib_path = os.path.join(model_dir, "reward_calibration.json")
        if os.path.exists(calib_path):
            with open(calib_path, "r") as f:
                obj = json.load(f)
            self.calib = obj.get("sorted_preds", None)

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
    def _score_single(self, text: str) -> float:
        chunks = self._encode_windows(text)
        preds = []; lens = []
        for ch in chunks:
            batch = self.tokenizer.prepare_for_model(
                ch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            # ★ ここで 1D -> 2D に正規化（BERT は [B, L] を期待）
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in batch and batch[k].dim() == 1:
                    batch[k] = batch[k].unsqueeze(0)

            lens.append(len(ch))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            preds.append(out.logits.squeeze(-1).detach().cpu().item())

        if not preds:
            # フォールバック：通常トークナイズでも 1D -> 2D を強制
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
            return out.logits.squeeze(-1).detach().cpu().item()

        if self.agg == "mean":
            return float(np.mean(preds))
        elif self.agg == "wmean":
            w = np.array(lens, dtype=float); w = w / (w.sum() + 1e-8)
            return float((w * np.array(preds)).sum())
        else:
            return float(np.median(preds))


    def _to_reward(self, raw: float) -> float:
        if self.calib:
            return _apply_percentile_map(raw, self.calib)
        return float(np.clip((raw - 1.0)/9.0, 0.0, 1.0))

    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        out = []
        for tx in texts:
            raw = self._score_single(tx)
            out.append({"raw_score": raw, "reward": self._to_reward(raw)})
        return out

    def score_ideas(self, titles: List[str], bodies: List[str]) -> List[Dict[str, float]]:
        texts = [(t or "").strip() + "\n\n" + (b or "").strip() for t, b in zip(titles, bodies)]
        return self.score_texts(texts)

# -----------------------------
# CORY 用：報酬関数
# -----------------------------
def irm_reward_fn(model_dir: str, idea_texts: List[str]) -> List[torch.Tensor]:
    scorer = IRMScorer(model_dir)
    results = scorer.score_texts(idea_texts)
    return [torch.tensor(x["reward"], dtype=torch.float32) for x in results]

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="IRMを学習する（.parquet / .json / .jsonl に対応）")
    p_train.add_argument("--data_path", required=True)
    p_train.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p_train.add_argument("--out", default="./irm_iclr_model")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--bs", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--max_len", type=int, default=512)
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--report_to", default="none", choices=["none", "wandb"])
    p_train.add_argument("--loss_type", default="huber", choices=["mse","mae","huber","rank"])
    p_train.add_argument("--rank_lambda", type=float, default=0.2)
    p_train.add_argument("--target_type", default="year_z", choices=["raw","year_z"])
    p_train.add_argument("--accept_threshold", type=float, default=6.0)
    p_train.add_argument("--add_year_tag", action="store_true")

    p_score = sub.add_parser("score", help="学習済みIRMでアイデアを採点（推論時スライディング対応）")
    p_score.add_argument("--model_dir", default="./irm_iclr_model")
    p_score.add_argument("--title", nargs="*", default=[])
    p_score.add_argument("--body",  nargs="*", default=[])
    p_score.add_argument("--max_len", type=int, default=512)

    p_asr = sub.add_parser("as_reward", help="CORY報酬として0〜1を返す")
    p_asr.add_argument("--model_dir", default="./irm_iclr_model")
    p_asr.add_argument("--text", nargs="*", default=[])
    p_asr.add_argument("--max_len", type=int, default=512)

    args = parser.parse_args()

    if args.cmd == "train":
        tc = TrainConfig(
            output_dir=args.out,
            model_name=args.model_name,
            max_length=args.max_len,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.bs,
            fp16=args.fp16,
            report_to=args.report_to,
            loss_type=args.loss_type,
            rank_lambda=args.rank_lambda,
            target_type=args.target_type,
            accept_threshold=args.accept_threshold,
            add_year_tag=args.add_year_tag,
        )
        train_irm(args.data_path, tc)

    elif args.cmd == "score":
        scorer = IRMScorer(args.model_dir, max_length=args.max_len)
        if not args.title and args.body:
            args.title = ["" for _ in args.body]
        if args.title and not args.body:
            args.body = ["" for _ in args.title]
        if len(args.title) != len(args.body):
            raise ValueError("title と body の個数を揃えてください")
        res = scorer.score_ideas(args.title, args.body)
        for i, r in enumerate(res):
            print(f"[{i}] raw_score={r['raw_score']:.3f}  reward={r['reward']:.3f}")

    elif args.cmd == "as_reward":
        ts = args.text
        if not ts:
            print("例) python irm_iclr.py as_reward --text 'New idea A...' 'New idea B...'")
        else:
            scorer = IRMScorer(args.model_dir, max_length=args.max_len)
            outs = scorer.score_texts(ts)
            for i, o in enumerate(outs):
                print(f"[{i}] reward={float(o['reward']):.3f}")
    else:
        parser.print_help()

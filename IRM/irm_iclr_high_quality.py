# irm_iclr_high_quality.py
# coding: utf-8
"""
改良版 IRM（Idea Reward Model）— 高品質・安定学習版
- 入力: ICLR系データ（.parquet / .json / .jsonl / ディレクトリ）
- 目的: タイトル+アブスト → スコア回帰（順序性/年差/ノイズを考慮した学習・報酬化）

主なポイント（前版からの改善）:
  1) DeBERTa v3 の tokenizer 警告への対処:
     - "deberta-v3" 系は `use_fast=False` デフォルトに（byte fallbackのUNK化を回避）
  2) 勾配チェックポイント安定化:
     - TrainingArguments(gradient_checkpointing_kwargs={"use_reentrant": False}) を追加
     - model.config.use_cache=False を厳密化
  3) Rank補助 loss の安定化と高速化:
     - ベクトル化（ペアの一括計算）＋ detach安全化（符号 s の計算は勾配外）
  4) ラベル/出力形状の厳格チェック:
     - uncertainty=True 時は (μ, logσ²)、False 時は (1次元) を強制
  5) 評価と保存:
     - best model の保存/復元条件を整備
     - Isotonic/分位キャリブの保存ロジック堅牢化
  6) 追加の実務オプション:
     - optimizer="adamw_torch_fused"（利用可能環境で高速/省メモリ）
     - lr_scheduler_type="cosine", warmup_ratio 既定値調整
  7) 推論側（IRMScorer）:
     - device 指定: 引数 device もしくは env IRM_DEVICE で "cuda:1" など選択可
     - 長文スライディングの境界条件を微修正
"""

import os, re, json, glob, math, random, inspect
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

# ---- Optional: isotonic calibration
_HAS_SK = True
try:
    from sklearn.isotonic import IsotonicRegression
    import joblib
except Exception:
    _HAS_SK = False

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

def _should_use_fast_tokenizer(model_name: str) -> bool:
    # DeBERTa v3 系は fast で byte fallback により UNK が増えるケースがあるため既定で False
    name = model_name.lower()
    if "deberta-v3" in name:
        return False
    return True

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
        mu = float(s.mean()); sd = float(s.std(ddof=0))
        stats[(None if k == -1 else int(k))] = (mu, sd if sd > 1e-8 else 1.0)
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
    # transformers のバージョン差に耐性
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

    year_tags = sorted({f"[YEAR={int(ex['year'])}]" for ex in examples if ex.get("year")})
    train_ex, valid_ex = stratified_split_by_score_bins(examples, cfg.train_ratio, cfg.seed)

    year_stats = default_stats = None
    if cfg.target_type == "year_z":
        year_stats, default_stats = compute_year_stats(train_ex)
        train_ex = z_transform_with_stats(train_ex, year_stats, default_stats)
        valid_ex = z_transform_with_stats(valid_ex, year_stats, default_stats)
    else:
        for ex in train_ex: ex["target"] = float(ex["score"])
        for ex in valid_ex: ex["target"] = float(ex["score"])

    ds_train = Dataset.from_list(train_ex)
    ds_valid = Dataset.from_list(valid_ex)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=_should_use_fast_tokenizer(cfg.model_name))
    if cfg.add_year_tag and year_tags:
        tokenizer.add_tokens(year_tags, special_tokens=False)

    def tok(batch):
        out = tokenizer(
            batch["text"], max_length=cfg.max_length,
            truncation=True, padding="max_length",
        )
        # labels を float32 できっちり
        out["labels"] = [float(x) for x in batch["target"]]
        out["score"]  = [float(x) for x in batch["score"]]
        out["accept"] = [int(x) for x in batch["accept"]]
        out["year"]   = [int(y) if (y is not None) else -1 for y in batch["year"]]
        return out

    dsd = DatasetDict({"train": ds_train, "validation": ds_valid})
    dsd = dsd.map(tok, batched=True, remove_columns=["text", "target", "score", "accept", "year"])
    dsd.set_format(type="torch")

    meta = {"year_stats": year_stats, "default_stats": default_stats, "year_tags": year_tags}
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
    tf32: bool = False
    gradient_checkpointing: bool = False
    loss_type: str = "huber"     # "mse" | "mae" | "huber" | "rank"
    rank_lambda: float = 0.2
    target_type: str = "year_z"  # "raw" | "year_z"
    accept_threshold: float = 6.0
    add_year_tag: bool = True
    uncertainty: bool = False     # [mu, logvar] で NLL
    use_isotonic: bool = False    # Isotonic校正保存/使用

def _spearman(preds: np.ndarray, labels: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(preds, labels).correlation)
    except Exception:
        return 0.0

class RankAuxTrainer(Trainer):
    def __init__(self, loss_type="huber", rank_lambda=0.2, use_uncertainty=False, **kwargs):
        super().__init__(**kwargs)
        self.loss_type = loss_type
        self.rank_lambda = rank_lambda
        self.use_uncertainty = use_uncertainty
        if loss_type == "huber":
            self._reg_loss = torch.nn.SmoothL1Loss()
        elif loss_type == "mae":
            self._reg_loss = torch.nn.L1Loss()
        else:
            self._reg_loss = torch.nn.MSELoss()

    def _pairwise_rank_loss(self, scores: torch.Tensor, labels: torch.Tensor, max_pairs: int = 64) -> torch.Tensor:
        # ランダムペアで比較（ベクトル化）
        bsz = scores.shape[0]
        if bsz < 2:
            return scores.new_zeros(())
        idx = torch.randperm(bsz, device=scores.device)
        i = idx[:max_pairs//2]
        j = idx[-(max_pairs//2):]
        i = i[:min(len(i), len(j))]
        j = j[:len(i)]
        if len(i) == 0:
            return scores.new_zeros(())
        # s = sign(y_i - y_j) は detach（ラベルの比較だけに使う）
        with torch.no_grad():
            s = torch.sign(labels[i] - labels[j])
        # 零のペアは除去
        mask = s != 0
        if mask.sum() == 0:
            return scores.new_zeros(())
        diff = scores[i[mask]] - scores[j[mask]]  # 勾配は scores にのみ流れる
        rank_loss = torch.nn.functional.softplus(-s[mask] * diff).mean()
        return rank_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # labels は自作 loss で用いる。モデルへ渡さない（内部 loss は無効化）
        labels = inputs.get("labels").float()
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits

        if self.use_uncertainty:
            # logits: [B,2] （mu, logvar）を想定。万一1次元ならフォールバック
            if logits.dim() == 1:
                logits = logits.unsqueeze(-1).repeat(1, 2)
            elif logits.size(-1) == 1:
                logits = torch.cat([logits, torch.zeros_like(logits)], dim=-1)
            mu = logits[..., 0]
            logvar = logits[..., 1].clamp(min=-10, max=10)
            # ガウスNLL
            nll = 0.5 * (logvar + (labels - mu) ** 2 / logvar.exp())
            reg = nll.mean()
            # rank 補助
            rank_loss = self._pairwise_rank_loss(mu, labels, max_pairs=64) if (self.loss_type == "rank" or self.rank_lambda > 0) else mu.new_zeros(())
            loss = reg + self.rank_lambda * rank_loss
        else:
            # 単一ユニット回帰
            logits = logits.squeeze(-1)
            if self.loss_type in ("huber", "mae", "mse"):
                reg = self._reg_loss(logits, labels)
                loss = reg
            elif self.loss_type == "rank":
                reg = torch.nn.functional.mse_loss(logits, labels)
                rank_loss = self._pairwise_rank_loss(logits, labels, max_pairs=64)
                loss = reg + self.rank_lambda * rank_loss
            else:
                loss = torch.nn.functional.mse_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss

def _build_model_for_regression(model_name: str, tokenizer, use_uncertainty: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=(1 if not use_uncertainty else 2),
        problem_type="regression",
        use_safetensors=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    # 勾配チェックポイント使用時の安定化
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model

def train_irm(data_path: str, tcfg: TrainConfig):
    torch.manual_seed(tcfg.seed); np.random.seed(tcfg.seed); random.seed(tcfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda, "matmul") and tcfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

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
    model = _build_model_for_regression(tcfg.model_name, tokenizer, tcfg.uncertainty)

    # bf16 が使えるなら bf16 優先（fp16より安定）
    bf16_flag = torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    args = _safe_training_args(
        output_dir=tcfg.output_dir,
        learning_rate=tcfg.lr,
        per_device_train_batch_size=tcfg.batch_size,
        per_device_eval_batch_size=max(4, min(16, tcfg.batch_size)),
        num_train_epochs=tcfg.epochs,
        weight_decay=tcfg.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        seed=tcfg.seed,
        fp16=(tcfg.fp16 and not bf16_flag),
        bf16=(bf16_flag if "bf16" in inspect.signature(TrainingArguments.__init__).parameters else False),
        warmup_ratio=tcfg.warmup_ratio,
        logging_steps=50,
        report_to=tcfg.report_to if tcfg.report_to in ("none", "wandb") else "none",
        gradient_accumulation_steps=1,
        gradient_checkpointing=tcfg.gradient_checkpointing,
        tf32=getattr(tcfg, "tf32", False),
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        gradient_checkpointing_kwargs={"use_reentrant": False}  # ★ PyTorch2系での再入可能版に起因する backward 二重実行対策
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple): preds = preds[0]
        preds = np.asarray(preds)
        # preds shape: [N] or [N,1] or [N,2]（uncertainty時）
        if preds.ndim == 1:
            mu = preds.astype(float)
        elif preds.ndim == 2:
            mu = preds[:, 0].astype(float)
        else:
            mu = preds.reshape(preds.shape[0], -1)[:, 0].astype(float)
        labels = labels.reshape(-1).astype(float)
        rmse = float(np.sqrt(np.mean((mu - labels) ** 2)))
        mae  = float(np.mean(np.abs(mu - labels)))
        spr  = _spearman(mu, labels)
        return {"rmse": rmse, "mae": mae, "spearman": spr}

    trainer = RankAuxTrainer(
        loss_type=tcfg.loss_type,
        rank_lambda=tcfg.rank_lambda,
        use_uncertainty=tcfg.uncertainty,
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,  # transformers>=5 は processing_class に移行
        compute_metrics=compute_metrics,
    )

    if tcfg.gradient_checkpointing and hasattr(trainer.model, "gradient_checkpointing_enable"):
        trainer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(trainer.model, "config"):
            trainer.model.config.use_cache = False

    trainer.train()
    trainer.save_model(tcfg.output_dir)
    tokenizer.save_pretrained(tcfg.output_dir)

    # year_z 統計保存
    if meta.get("year_stats") is not None:
        ys = meta["year_stats"]; ds = meta["default_stats"]
        stats_obj = {
            "year_stats": { ("" if k is None else str(k)): {"mean": float(v[0]), "std": float(v[1])} for k, v in ys.items() },
            "default": {"mean": float(ds[0]), "std": float(ds[1])},
            "target_type": tcfg.target_type,
        }
        with open(os.path.join(tcfg.output_dir, "year_z_stats.json"), "w") as f:
            json.dump(stats_obj, f)

    # 校正（isotonic or percentile）— validation 予測に基づく
    eval_out = trainer.predict(dsd["validation"])
    val_preds_raw = eval_out.predictions
    val_preds_raw = np.asarray(val_preds_raw)
    if val_preds_raw.ndim == 1:
        mu_preds = val_preds_raw.astype(float)
    elif val_preds_raw.ndim == 2:
        mu_preds = val_preds_raw[:, 0].astype(float)
    else:
        mu_preds = val_preds_raw.reshape(val_preds_raw.shape[0], -1)[:, 0].astype(float)

    if tcfg.use_isotonic and _HAS_SK:
        order = np.argsort(mu_preds)
        ranks = np.empty_like(order, dtype=float); ranks[order] = np.linspace(0.0, 1.0, len(mu_preds))
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True)
        iso.fit(mu_preds, ranks)
        joblib.dump(iso, os.path.join(tcfg.output_dir, "reward_iso.joblib"))
        print("[INFO] Saved isotonic calibration.")
    else:
        calib_sorted = np.sort(mu_preds).tolist()
        with open(os.path.join(tcfg.output_dir, "reward_calibration.json"), "w") as f:
            json.dump({"sorted_preds": calib_sorted}, f)
        if tcfg.use_isotonic and not _HAS_SK:
            print("[WARN] scikit-learn/joblib が無いため Isotonic をスキップし、分位キャリブを保存しました。")
        else:
            print("[INFO] Saved percentile calibration.")

    print("[INFO] Saved calibration and year_z_stats (if any).")

# -----------------------------
# 推論（スコアリング + スライディング）
# -----------------------------
def _apply_percentile_map(x: float, sorted_preds: List[float]) -> float:
    if not sorted_preds: return float(np.clip((x - 1.0)/9.0, 0.0, 1.0))
    import bisect as _bisect
    idx = _bisect.bisect_left(sorted_preds, x)
    n = len(sorted_preds)
    return float(np.clip((idx / (n - 1)) if n > 1 else 0.5, 0.0, 1.0))

class IRMScorer:
    """
    - raw_score: 回帰モデル出力（uncertainty有効時は μ を返す）
    - reward: Isotonic か 分位キャリブ で [0,1] に校正
    - 推論時スライディング（stride率・集約方式を選択）
    """
    def __init__(self, model_dir: str, max_length: int = 512,
                 window_stride_ratio: float = 0.5, agg: str = "median",
                 device: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=_should_use_fast_tokenizer(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_safetensors=True)
        self.max_length = max_length
        self.stride = max(1, int(max_length * window_stride_ratio))
        self.agg = agg  # "mean" | "median" | "wmean"
        # device 選択: 引数 > 環境変数 IRM_DEVICE > CUDA自動
        env_dev = os.environ.get("IRM_DEVICE", None)
        dev = device or env_dev
        if dev:
            self.device = torch.device(dev)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device); self.model.eval()
        self.calib = None; self.iso = None
        iso_path = os.path.join(model_dir, "reward_iso.joblib")
        calib_path = os.path.join(model_dir, "reward_calibration.json")
        if os.path.exists(iso_path) and _HAS_SK:
            try: self.iso = joblib.load(iso_path)
            except Exception: self.iso = None
        if os.path.exists(calib_path):
            try:
                with open(calib_path, "r") as f:
                    obj = json.load(f)
                self.calib = obj.get("sorted_preds", None)
            except Exception:
                self.calib = None

    def _encode_windows(self, text: str):
        # fast=False でも等価に動くため自前スライスで安定化
        ids = self.tokenizer(text, truncation=False, padding=False, return_tensors=None)["input_ids"]
        chunks = []; start = 0
        n = len(ids)
        if n == 0:
            return [[]]
        while start < n:
            end = min(start + self.max_length, n)
            ch = ids[start:end]
            chunks.append(ch)
            if end >= n:
                break
            # オーバーラップ移動
            start = max(start + self.max_length - self.stride, 0)
            if start >= n:
                break
        return chunks

    @torch.no_grad()
    def _score_single(self, text: str) -> Tuple[float, Optional[float]]:
        chunks = self._encode_windows(text)
        preds = []; lens = []; logvars = []

        def _extract_mu_var(logits_tensor):
            # logits_tensor: [1, C] を想定
            if logits_tensor.dim() == 1:
                logits_tensor = logits_tensor.unsqueeze(0)
            if logits_tensor.size(-1) == 1:
                mu = logits_tensor[..., 0]
                logvar = None
            else:
                mu = logits_tensor[..., 0]
                logvar = logits_tensor[..., 1].clamp(min=-10, max=10)
            return mu, logvar

        for ch in chunks:
            batch = self.tokenizer.prepare_for_model(
                ch, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
            )
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in batch and batch[k].dim() == 1: batch[k] = batch[k].unsqueeze(0)
            lens.append(len(ch))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            mu, logvar = _extract_mu_var(out.logits.detach())
            preds.append(float(mu.squeeze(0).cpu()))
            if logvar is not None:
                logvars.append(float(logvar.squeeze(0).cpu()))

        if not preds:
            batch = self.tokenizer(text, truncation=True, padding="max_length",
                                   max_length=self.max_length, return_tensors="pt")
            for k in ("input_ids", "attention_mask", "token_type_ids"):
                if k in batch and batch[k].dim() == 1: batch[k] = batch[k].unsqueeze(0)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            mu, logvar = _extract_mu_var(out.logits.detach())
            mu_val = float(mu.squeeze(0).cpu())
            unc = float(math.exp(0.5 * float(logvar.squeeze(0).cpu()))) if logvar is not None else None
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
                if uncertainty is not None:
                    r = r * float(1.0 / (1.0 + 0.1 * uncertainty))
                return float(np.clip(r, 0.0, 1.0))
            except Exception:
                pass
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
        out = []
        for tx in texts:
            raw, unc = self._score_single(tx)
            out.append({"raw_score": raw, "reward": self._to_reward(raw, unc),
                        "uncertainty": (float(unc) if unc is not None else None)})
        return out

    def score_ideas(self, titles: List[str], bodies: List[str]) -> List[Dict[str, float]]:
        texts = [(t or "").strip() + "\n\n" + (b or "").strip() for t, b in zip(titles, bodies)]
        return self.score_texts(texts)

# -----------------------------
# CORY 用：報酬関数
# -----------------------------
def irm_reward_fn(model_dir: str, idea_texts: List[str], device: Optional[str] = None) -> List[torch.Tensor]:
    scorer = IRMScorer(model_dir, device=device)
    results = scorer.score_texts(idea_texts)
    return [torch.tensor(x["reward"], dtype=torch.float32) for x in results]

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="IRMを学習（.parquet / .json / .jsonl / ディレクトリ）")
    p_train.add_argument("--data_path", required=True)
    p_train.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    p_train.add_argument("--out", default="./irm_iclr_model")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--bs", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--max_len", type=int, default=512)
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--tf32", action="store_true")
    p_train.add_argument("--gc", dest="gradient_checkpointing", action="store_true")
    p_train.add_argument("--report_to", default="none", choices=["none", "wandb"])
    p_train.add_argument("--loss_type", default="huber", choices=["mse","mae","huber","rank"])
    p_train.add_argument("--rank_lambda", type=float, default=0.2)
    p_train.add_argument("--target_type", default="year_z", choices=["raw","year_z"])
    p_train.add_argument("--accept_threshold", type=float, default=6.0)
    p_train.add_argument("--add_year_tag", action="store_true")
    p_train.add_argument("--uncertainty", action="store_true")
    p_train.add_argument("--use_isotonic", action="store_true")

    p_score = sub.add_parser("score", help="学習済みIRMで採点（スライディング対応）")
    p_score.add_argument("--model_dir", default="./irm_iclr_model")
    p_score.add_argument("--title", nargs="*", default=[])
    p_score.add_argument("--body",  nargs="*", default=[])
    p_score.add_argument("--max_len", type=int, default=512)
    p_score.add_argument("--device", default=None, help="例: cuda:1 / cuda / cpu")

    p_asr = sub.add_parser("as_reward", help="CORY報酬として0〜1を返す")
    p_asr.add_argument("--model_dir", default="./irm_iclr_model")
    p_asr.add_argument("--text", nargs="*", default=[])
    p_asr.add_argument("--max_len", type=int, default=512)
    p_asr.add_argument("--device", default=None, help="例: cuda:1 / cuda / cpu")

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
            tf32=args.tf32,
            gradient_checkpointing=args.gradient_checkpointing,
            report_to=args.report_to,
            loss_type=args.loss_type,
            rank_lambda=args.rank_lambda,
            target_type=args.target_type,
            accept_threshold=args.accept_threshold,
            add_year_tag=args.add_year_tag,
            uncertainty=args.uncertainty,
            use_isotonic=args.use_isotonic,
        )
        train_irm(args.data_path, tc)

    elif args.cmd == "score":
        scorer = IRMScorer(args.model_dir, max_length=args.max_len, device=args.device)
        if not args.title and args.body:
            args.title = ["" for _ in args.body]
        if args.title and not args.body:
            args.body = ["" for _ in args.title]
        if len(args.title) != len(args.body):
            raise ValueError("title と body の個数を揃えてください")
        res = scorer.score_ideas(args.title, args.body)
        for i, r in enumerate(res):
            unc_str = f"  unc={r['uncertainty']:.3f}" if r["uncertainty"] is not None else ""
            print(f"[{i}] raw_score={r['raw_score']:.3f}  reward={r['reward']:.3f}{unc_str}")

    elif args.cmd == "as_reward":
        ts = args.text
        if not ts:
            print("例) python irm_iclr_high_quality.py as_reward --text 'New idea A...' 'New idea B...'")
        else:
            scorer = IRMScorer(args.model_dir, max_length=args.max_len, device=args.device)
            outs = scorer.score_texts(ts)
            for i, o in enumerate(outs):
                if o["uncertainty"] is not None:
                    print(f"[{i}] reward={float(o['reward']):.3f}  unc={float(o['uncertainty']):.3f}")
                else:
                    print(f"[{i}] reward={float(o['reward']):.3f}")
    else:
        parser.print_help()
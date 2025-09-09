# irm_iclr.py
# coding: utf-8
"""
ICLRデータ（berenslab/iclr-dataset の .parquet など）を用いた IRM（Idea Reward Model）
- 回帰モデル：タイトル+アブストラクト → 予測ICLRスコア（概ね1〜10）
- 報酬：0〜1に線形正規化（CORYのPPO報酬としてそのまま利用可能）
- 入力フォーマット：.parquet / .json / .jsonl（再帰探索）
- 主要依存：transformers, datasets, torch, pandas (pyarrow), numpy
"""

import os
import re
import json
import glob
import math
import random
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
import pyarrow as pa

# -----------------------------
# 0) ユーティリティ
# -----------------------------
RATING_PAT = re.compile(r"^\s*(\d+(\.\d+)?)")  # "6: Weak Accept" → 6, "7" → 7

def extract_numeric_rating(text: str) -> Optional[float]:
    """ "6: Weak Accept" → 6.0 のように先頭数値を抽出 """
    if text is None:
        return None
    m = RATING_PAT.match(str(text))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

# -----------------------------
# 1) ローダ（JSON/JSONL）
# -----------------------------
def load_iclr_records_from_dir(data_dir: str) -> List[Dict]:
    """
    data_dir 以下の *.json / *.jsonl をすべて読み込み、dictのリストに。
    期待スキーマ（柔軟に対応）:
      { "title": str, "abstract": str, "reviews": [{"rating": "6: Weak Accept", ...}, ...] }
    """
    paths = []
    paths += glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    paths += glob.glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)

    records: List[Dict] = []
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
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

# -----------------------------
# 2) ローダ（Parquet）
# -----------------------------
# 先頭に追加/確認
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

def load_iclr_parquet(parquet_path: str) -> List[Dict]:
    """
    berenslab/iclr-dataset の .parquet を PyArrow だけで読み、
    pandas 経由にせず Python リスト化して安全に取り出す版。
    期待列（年版で揺れる可能性あり）:
      - title
      - abstract
      - scores / score / ratings / review_scores / scores_mean / avg_score
    """
    pf = pq.ParquetFile(parquet_path)

    # 列名一覧を取得
    schema = pf.schema_arrow
    cols = schema.names

    # 列名の候補
    title_col = "title"
    abs_candidates = ["abstract", "abstract_text", "paper_abstract"]
    score_candidates = ["scores", "score", "ratings", "review_scores", "scores_mean", "avg_score"]

    # 実在する列で確定
    abs_col = next((c for c in abs_candidates if c in cols), None)
    score_col = next((c for c in score_candidates if c in cols), None)
    if abs_col is None:
        raise RuntimeError(f"{parquet_path} に abstract 列が見つかりません。列: {cols}")
    if score_col is None:
        raise RuntimeError(f"{parquet_path} にスコア列が見つかりません。列: {cols}")
    need_cols = [c for c in [title_col, abs_col, score_col] if c in cols]

    # pandas メタデータを完全に無視して読み出し（→ Table）
    # ※ use_pandas_metadata=False を明示しておくと安心（PyArrow 14+）
    table = pf.read(columns=need_cols, use_pandas_metadata=False)

    # 各列を Python リスト化（pandas を経由しない）
    def col_list(name: str) -> List:
        if name not in need_cols:
            # 列がない（titleが存在しない古い/特殊版など）→ 空で返す
            return ["" for _ in range(table.num_rows)]
        arr = table.column(name)  # ChunkedArray
        # ListArray / LargeList でも .to_pylist() で Python ネイティブへ
        return arr.to_pylist()

    titles = col_list(title_col) if title_col in need_cols else ["" for _ in range(table.num_rows)]
    abstracts = col_list(abs_col)
    scores_raw = col_list(score_col)

    recs: List[Dict] = []
    for t, a, sv in zip(titles, abstracts, scores_raw):
        title = str(t or "").strip()
        abstract = str(a or "").strip()

        # スコアの型に応じて平均化/数値化
        score = None
        try:
            if isinstance(sv, (list, tuple, np.ndarray)):
                arr = [float(x) for x in sv if x is not None and not (isinstance(x, float) and np.isnan(x))]
                if arr:
                    score = float(np.mean(arr))
            elif isinstance(sv, str):
                # "6: Weak Accept" など → 先頭数値を抽出
                num = extract_numeric_rating(sv)
                if num is not None:
                    score = float(num)
            elif sv is not None and not (isinstance(sv, float) and np.isnan(sv)):
                score = float(sv)
        except Exception:
            pass

        recs.append({
            "title": title,
            "abstract": abstract,
            "score": score,  # None は後段で除外
        })
    return recs


# -----------------------------
# 3) データセット化
# -----------------------------
@dataclass
class DataConfig:
    data_path: str               # ディレクトリ or ファイル（.parquet / .json / .jsonl）
    model_name: str = "distilroberta-base"
    max_length: int = 512
    seed: int = 123
    train_ratio: float = 0.9

def make_examples(data_path: str, min_len: int = 40) -> List[Dict]:
    """
    data_path がフォルダなら parquet優先で1つ選択、無ければ JSON/JSONL を再帰探索。
    data_path がファイルなら拡張子で分岐して読み込み、共通形式の examples に整形。
    """
    records: List[Dict] = []

    if os.path.isdir(data_path):
        # ディレクトリ：parquet優先
        pq_files = glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True)
        if pq_files:
            # 年次が新しそうなものを優先（簡易：ファイル名降順）
            pq_files.sort(reverse=True)
            print(">> Using parquet:", pq_files[0])
            records = load_iclr_parquet(pq_files[0])
        else:
            records = load_iclr_records_from_dir(data_path)
    else:
        # 単一ファイル
        if data_path.endswith(".parquet"):
            records = load_iclr_parquet(data_path)
        elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
            base_dir = os.path.dirname(data_path) or "."
            # 単一ファイルも読むが、同ディレクトリの他JSONも拾いたい場合はここを調整
            # ここでは単体読み込みに限定
            if data_path.endswith(".jsonl"):
                with open(data_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                pass
            else:
                try:
                    with open(data_path, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if isinstance(obj, list):
                        records.extend(obj)
                    else:
                        records.append(obj)
                except Exception:
                    pass
        else:
            raise ValueError("data_path はディレクトリまたは .parquet/.json/.jsonl を指定してください。")

    # records → examples へ
    examples: List[Dict] = []
    for rec in records:
        title = (rec.get("title") or "").strip()
        abs_  = (rec.get("abstract") or rec.get("abstract_text") or "").strip()
        text = (title + "\n\n" + abs_).strip()
        if len(text) < min_len:
            continue

        score = rec.get("score", None)
        if score is None:
            # JSON派生データ（reviews配列内の rating を平均化）
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
        if score is None:
            continue

        examples.append({"text": text, "score": float(score)})

    return examples

def make_dataset(cfg: DataConfig) -> Tuple[DatasetDict, AutoTokenizer]:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    examples = make_examples(cfg.data_path)
    if len(examples) < 100:
        raise RuntimeError(f"学習サンプルが少なすぎます（{len(examples)}件）。データ/列名を確認してください。")

    random.shuffle(examples)
    n_train = int(len(examples) * cfg.train_ratio)
    ds_train = Dataset.from_list(examples[:n_train])
    ds_valid = Dataset.from_list(examples[n_train:])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    def tok(batch):
        out = tokenizer(
            batch["text"],
            max_length=cfg.max_length,
            truncation=True,
            padding="max_length",
        )
        # ★ ここを変更：NumPy配列ではなく Python リストで返す
        # out["labels"] = np.array(batch["score"], dtype=np.float32)
        out["labels"] = [float(x) for x in batch["score"]]  # または np.asarray(..., dtype=np.float32).tolist()
        return out

    dsd = DatasetDict({"train": ds_train, "validation": ds_valid})
    dsd = dsd.map(tok, batched=True, remove_columns=["text", "score"])
    dsd.set_format(type="torch")
    return dsd, tokenizer

# -----------------------------
# 4) 学習
# -----------------------------
@dataclass
class TrainConfig:
    output_dir: str = "./irm_iclr_model"
    model_name: str = "distilroberta-base"
    max_length: int = 512
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    weight_decay: float = 0.01
    seed: int = 123
    warmup_ratio: float = 0.06
    fp16: bool = True
    report_to: str = "none"  # "wandb" などにしたい場合は変更

def _spearman(preds: np.ndarray, labels: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(preds, labels).correlation)
    except Exception:
        # scipy未導入なら0.0を返す
        return 0.0

def train_irm(data_path: str, tcfg: TrainConfig):
    dcfg = DataConfig(
        data_path=data_path,
        model_name=tcfg.model_name,
        max_length=tcfg.max_length,
        seed=tcfg.seed,
    )
    dsd, tokenizer = make_dataset(dcfg)

    model = AutoModelForSequenceClassification.from_pretrained(
        tcfg.model_name,
        num_labels=1,                     # ← 回帰
        problem_type="regression"
    )

    args = TrainingArguments(
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
        fp16=tcfg.fp16,
        warmup_ratio=tcfg.warmup_ratio,
        logging_steps=50,
        report_to=tcfg.report_to,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = preds.reshape(-1)
        rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
        mae  = float(np.mean(np.abs(preds - labels)))
        spr  = _spearman(preds, labels)
        return {"rmse": rmse, "mae": mae, "spearman": spr}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dsd["train"],
        eval_dataset=dsd["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(tcfg.output_dir)
    tokenizer.save_pretrained(tcfg.output_dir)

# -----------------------------
# 5) 推論（スコアリング）
# -----------------------------
class IRMScorer:
    """
    学習済みIRMを読み込み、テキストにスコアを付ける。
    - raw_score: 回帰モデルの予測（目安：1〜10）
    - reward: 0〜1に線形正規化（clip）
    """
    def __init__(self, model_dir: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_texts(self, texts: List[str]) -> List[Dict[str, float]]:
        batch = self.tokenizer(
            texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt"
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(**batch)
        preds = outputs.logits.squeeze(-1).detach().cpu().numpy().tolist()

        out: List[Dict[str, float]] = []
        for p in preds:
            raw = float(p)
            # 1〜10 を想定した線形写像（外れ値はclip）
            reward = float(np.clip((raw - 1.0) / 9.0, 0.0, 1.0))
            out.append({"raw_score": raw, "reward": reward})
        return out

    def score_ideas(self, titles: List[str], bodies: List[str]) -> List[Dict[str, float]]:
        texts = [(t or "").strip() + "\n\n" + (b or "").strip() for t, b in zip(titles, bodies)]
        return self.score_texts(texts)

# -----------------------------
# 6) CORY 用：報酬関数
# -----------------------------
def irm_reward_fn(model_dir: str, idea_texts: List[str]) -> List[torch.Tensor]:
    """
    CORY学習ループで呼ぶ：
        rewards = irm_reward_fn("./irm_iclr_model", [q + r for q, r in zip(q_txt, resp)])
    返り値：List[torch.tensor(float32)]（0〜1）
    """
    scorer = IRMScorer(model_dir)
    results = scorer.score_texts(idea_texts)
    return [torch.tensor(x["reward"], dtype=torch.float32) for x in results]

# -----------------------------
# 7) CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_train = sub.add_parser("train", help="IRMを学習する（.parquet / .json / .jsonl に対応）")
    p_train.add_argument("--data_path", required=True, help="フォルダ or ファイルパス")
    p_train.add_argument("--model_name", default="distilroberta-base")
    p_train.add_argument("--out", default="./irm_iclr_model")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--bs", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--max_len", type=int, default=512)
    p_train.add_argument("--fp16", action="store_true")
    p_train.add_argument("--report_to", default="none", choices=["none", "wandb"])

    p_score = sub.add_parser("score", help="学習済みIRMでアイデア（title/body）を採点")
    p_score.add_argument("--model_dir", default="./irm_iclr_model")
    p_score.add_argument("--title", nargs="*", default=[])
    p_score.add_argument("--body",  nargs="*", default=[])

    p_asr = sub.add_parser("as_reward", help="CORY報酬として0〜1を返す")
    p_asr.add_argument("--model_dir", default="./irm_iclr_model")
    p_asr.add_argument("--text", nargs="*", default=[])

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
        )
        train_irm(args.data_path, tc)

    elif args.cmd == "score":
        scorer = IRMScorer(args.model_dir)
        # 片側だけ指定された場合は空文字で揃える
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
            print("例) python irm_iclr.py as_reward --text 'New self-supervised curation method...' 'A differentiable bisociation module...'")
        else:
            tens = irm_reward_fn(args.model_dir, ts)
            for i, t in enumerate(tens):
                print(f"[{i}] reward={float(t):.3f}")
    else:
        parser.print_help()

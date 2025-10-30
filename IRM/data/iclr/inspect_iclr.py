# inspect_iclr.py
# coding: utf-8
"""
ICLR系 .parquet / JSON / JSONL の中身を詳細点検するスクリプト
- スキーマ（列名）と候補列の自動検出
- 年別の件数・スコア統計・accept率（閾値指定可）
- 欠損率チェック
- スコア分布の要約（全体＆年別）
- 見本レコード（タイトル/アブストの冒頭）
- （任意）トークン長の概算 (transformers tokenizer)

使い方:
  python inspect_iclr.py --path data/iclr/iclr25v2.parquet --accept_threshold 6.0 --samples_per_year 3 --show_text
  python inspect_iclr.py --path data/iclr/dir_with_jsons --accept_threshold 6.0

オプション:
  --token_stats         : SciBERTトークナイザで入力長を概算（時間が少しかかります）
  --model_name          : トークン長の計算に使うモデル（既定: allenai/scibert_scivocab_uncased）
  --max_preview_chars   : タイトル/アブスト表示の最大文字数（既定: 140）
  --samples_per_year    : 年ごとに表示するサンプル数（既定: 2）
  --accept_threshold    : acceptとみなすスコアの閾値（既定: 6.0）
  --show_text           : サンプルのタイトル/アブストを表示する
"""

import os, re, json, glob, math, argparse, statistics
from typing import List, Dict, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# -------- 候補列名（データセット差吸収） --------
TITLE_COL = "title"
ABS_CANDS = ["abstract", "abstract_text", "paper_abstract", "abstractText"]
SCORE_CANDS = ["scores", "score", "ratings", "review_scores", "scores_mean", "avg_score"]
YEAR_CANDS = ["year", "conference_year", "submission_year"]
DECISION_CANDS = ["decision", "final_decision", "accept", "accepted"]

RATING_PAT = re.compile(r"^\s*(\d+(\.\d+)?)")

def extract_numeric_rating(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            if math.isnan(x): return None
        except Exception:
            pass
        return float(x)
    s = str(x)
    m = RATING_PAT.match(s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def list_files(path: str) -> List[str]:
    if os.path.isdir(path):
        fs = glob.glob(os.path.join(path, "**/*.*"), recursive=True)
        return [f for f in fs if os.path.isfile(f) and os.path.splitext(f)[1].lower() in (".parquet", ".json", ".jsonl")]
    else:
        return [path]

def read_parquet_columns(pq_path: str, need_cols: List[str]) -> Dict[str, List]:
    pf = pq.ParquetFile(pq_path)
    cols_avail = [c for c in need_cols if c in pf.schema_arrow.names]
    tbl = pf.read(columns=cols_avail, use_pandas_metadata=False)
    out = {}
    for c in need_cols:
        if c in cols_avail:
            out[c] = tbl.column(c).to_pylist()
        else:
            out[c] = [None] * tbl.num_rows
    return out

def read_records(path: str) -> List[Dict]:
    recs: List[Dict] = []
    files = list_files(path)
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext == ".parquet":
            # まずスキーマを見て列を決める
            pf = pq.ParquetFile(f)
            cols = set(pf.schema_arrow.names)
            abs_col = next((c for c in ABS_CANDS if c in cols), None)
            score_col = next((c for c in SCORE_CANDS if c in cols), None)
            year_col = next((c for c in YEAR_CANDS if c in cols), None)
            dec_col = next((c for c in DECISION_CANDS if c in cols), None)
            need = [TITLE_COL]
            for c in (abs_col, score_col, year_col, dec_col):
                if c: need.append(c)

            data = read_parquet_columns(f, need)
            titles = data.get(TITLE_COL, [])
            abstracts = data.get(abs_col, []) if abs_col else [None] * len(titles)
            scores_raw = data.get(score_col, []) if score_col else [None] * len(titles)
            years = data.get(year_col, []) if year_col else [None] * len(titles)
            decis = data.get(dec_col, []) if dec_col else [None] * len(titles)

            for t, a, s, y, d in zip(titles, abstracts, scores_raw, years, decis):
                # スコア正規化
                sc = None
                try:
                    if isinstance(s, (list, tuple, np.ndarray)):
                        arr = [extract_numeric_rating(x) for x in s]
                        arr = [x for x in arr if x is not None]
                        if arr:
                            sc = float(np.mean(arr))
                    else:
                        sc = extract_numeric_rating(s)
                except Exception:
                    sc = None

                # 年正規化
                yy = None
                if y is not None:
                    try:
                        ystr = str(y)
                        m = re.findall(r"\d{4}", ystr)
                        if m: yy = int(m[0])
                    except Exception:
                        yy = None

                # decision 下処理
                dd = None
                if isinstance(d, str):
                    dd = d.strip().lower()
                elif isinstance(d, (int, float)):
                    dd = "accept" if d else "reject"

                recs.append({
                    "title": (t or ""),
                    "abstract": (a or ""),
                    "score": sc,
                    "year": yy,
                    "decision": dd,
                    "_source_file": os.path.basename(f),
                })

        elif ext in (".json", ".jsonl"):
            # JSON/JSONL もざっくり読む（年やスコアは上と同様のルールで抽出）
            if ext == ".jsonl":
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line: continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        recs.append(normalize_json_record(obj, f))
            else:
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        obj = json.load(fh)
                    if isinstance(obj, list):
                        for o in obj:
                            recs.append(normalize_json_record(o, f))
                    else:
                        recs.append(normalize_json_record(obj, f))
                except Exception:
                    pass
    return recs

def normalize_json_record(obj: Dict, src: str) -> Dict:
    # タイトルとアブスト
    title = obj.get("title") or ""
    abs_ = obj.get("abstract") or obj.get("abstract_text") or obj.get("paper_abstract") or ""

    # スコア候補
    sc = obj.get("score")
    if sc is None:
        # reviews に ratings がある場合
        reviews = obj.get("reviews") or obj.get("meta_reviews") or []
        if isinstance(reviews, dict): reviews = [reviews]
        arr = []
        for rv in reviews:
            cand = rv.get("rating") or rv.get("final_decision") or rv.get("recommendation")
            num = extract_numeric_rating(cand)
            if num is not None: arr.append(num)
        if arr:
            sc = float(np.mean(arr))
    else:
        sc = extract_numeric_rating(sc)

    # 年候補
    yy = obj.get("year") or obj.get("conference_year") or obj.get("submission_year")
    year = None
    if yy is not None:
        try:
            ystr = str(yy)
            m = re.findall(r"\d{4}", ystr)
            if m: year = int(m[0])
        except Exception:
            year = None

    # decision
    d = obj.get("decision") or obj.get("final_decision") or obj.get("accept")
    dd = None
    if isinstance(d, str):
        dd = d.strip().lower()
    elif isinstance(d, (int, float)):
        dd = "accept" if d else "reject"

    return {
        "title": str(title),
        "abstract": str(abs_),
        "score": sc,
        "year": year,
        "decision": dd,
        "_source_file": os.path.basename(src),
    }

def pct(x, p):
    if not x: return None
    return float(np.quantile(np.array(x, dtype=float), p))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="parquet or dir (parquet/json/jsonl)")
    ap.add_argument("--accept_threshold", type=float, default=6.0)
    ap.add_argument("--samples_per_year", type=int, default=2)
    ap.add_argument("--show_text", action="store_true")
    ap.add_argument("--token_stats", action="store_true")
    ap.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--max_preview_chars", type=int, default=140)
    args = ap.parse_args()

    recs = read_records(args.path)
    n = len(recs)
    print(f"\n=== Loaded {n} records from: {args.path} ===")
    if n == 0:
        return

    # スキーマ風サマリ
    keys = set()
    for r in recs:
        keys.update(list(r.keys()))
    print("\n[Detected fields]")
    print(sorted(keys))

    # 欠損率
    def miss_rate(field):
        miss = sum(1 for r in recs if (r.get(field) is None or r.get(field) == ""))
        return miss / n
    print("\n[Missing ratios]")
    for f in ["title", "abstract", "score", "year", "decision"]:
        print(f"- {f:8s}: {miss_rate(f)*100:.2f}%")

    # 全体統計（スコア）
    scores = [r["score"] for r in recs if r["score"] is not None]
    years = [r["year"] for r in recs if r["year"] is not None]
    y_min, y_max = (min(years) if years else None, max(years) if years else None)
    print("\n[Score (overall)]")
    if scores:
        print(f"count={len(scores)}, mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, "
              f"min={np.min(scores):.3f}, p25={pct(scores,0.25):.3f}, "
              f"median={pct(scores,0.5):.3f}, p75={pct(scores,0.75):.3f}, max={np.max(scores):.3f}")
    else:
        print("no scores detected")

    print("\n[Years present]")
    if years:
        # 件数 by year
        by_year_counts = {}
        for r in recs:
            y = r["year"]
            if y is None: continue
            by_year_counts[y] = by_year_counts.get(y, 0) + 1
        for y in sorted(by_year_counts.keys()):
            print(f"- {y}: {by_year_counts[y]} records")
        print(f"year range: {y_min}..{y_max}")
    else:
        print("no year detected")

    # 年別統計
    print("\n[Per-year stats]")
    per = {}
    for r in recs:
        y = r["year"]
        if y is None: continue
        per.setdefault(y, []).append(r)
    if not per:
        print("(no year)")
    else:
        header = "year    n    score_mean  score_std   min   p25  median  p75   max   accept@{th}  accept_ratio".format(th=args.accept_threshold)
        print(header)
        for y in sorted(per.keys()):
            vv = per[y]
            scs = [v["score"] for v in vv if v["score"] is not None]
            if scs:
                acc = sum(1 for s in scs if s >= args.accept_threshold)
                print(f"{y:<7d}{len(vv):<5d}{np.mean(scs):>12.3f}{np.std(scs):>11.3f}"
                      f"{np.min(scs):>7.1f}{pct(scs,0.25):>6.1f}{pct(scs,0.5):>8.1f}"
                      f"{pct(scs,0.75):>6.1f}{np.max(scs):>6.1f}{acc:>11d}{(acc/len(scs))*100:>13.2f}%")
            else:
                print(f"{y:<7d}{len(vv):<5d}(no score)")

    # サンプル表示
    if args.show_text:
        print("\n[Samples per year]")
        for y in sorted(per.keys()):
            print(f"\n--- Year {y} ---")
            shown = 0
            for r in per[y]:
                if shown >= args.samples_per_year: break
                title = (r.get("title") or "").strip().replace("\n"," ")
                abstract = (r.get("abstract") or "").strip().replace("\n"," ")
                if len(title) > args.max_preview_chars:
                    title = title[:args.max_preview_chars] + "..."
                if len(abstract) > args.max_preview_chars:
                    abstract = abstract[:args.max_preview_chars] + "..."
                print(f"* score={r.get('score')}, file={r.get('_source_file')}")
                print(f"  - title   : {title}")
                print(f"  - abstract: {abstract}")
                shown += 1

    # （任意）トークン長の概算
    if args.token_stats:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
            lens = []
            for r in recs:
                txt = f"{r.get('title') or ''}\n\n{r.get('abstract') or ''}"
                ids = tok(txt, truncation=False, padding=False, return_tensors=None)["input_ids"]
                lens.append(len(ids))
            print("\n[Token length stats]")
            print(f"count={len(lens)}, mean={np.mean(lens):.1f}, std={np.std(lens):.1f}, "
                  f"min={np.min(lens)}, p25={int(pct(lens,0.25))}, median={int(pct(lens,0.5))}, "
                  f"p75={int(pct(lens,0.75))}, max={np.max(lens)}")
        except Exception as e:
            print("\n[Token length stats] skipped:", e)

if __name__ == "__main__":
    main()


#　使い方
# python inspect_iclr.py --path ./iclr25v2.parquet
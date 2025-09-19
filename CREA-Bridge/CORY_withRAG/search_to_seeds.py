#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
search_to_seeds.py
- FAISS索引 + メタデータ + クエリ群 から「研究シード」を作成
- 出力: ./data/research_seeds.jsonl （各行が1シード）
- シード形式:
  {
    "topic": "<クエリ文字列>",
    "problem": "<代表論文タイトル>",
    "sources": ["vision","robotics","LLM"],     # meta.fields を集約
    "constraints": "real-world viability",      # 任意文字列（コマンド引数で指定可）
    "context": [
      {"title":"...", "abstract":"...", "fields":[...]},
      ...
    ]
  }

使い方:
    python search_to_seeds.py \
      --queries ./data/queries.txt \
      --index ./rag/index.faiss \
      --meta ./rag/meta.jsonl \
      --out ./data/research_seeds.jsonl \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --k 5 --mmr_lambda 0.6 \
      --constraints "real-world viability"
依存:
    pip install sentence-transformers faiss-cpu tqdm
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer


def load_meta(meta_path: str) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            metas.append(json.loads(ln))
    if not metas:
        raise ValueError(f"no meta found at {meta_path}")
    return metas


def load_queries(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        qs = [ln.strip() for ln in f if ln.strip()]
    if not qs:
        raise ValueError(f"no queries in {path}")
    return qs


def mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    top_k: int = 5,
    lambda_: float = 0.6,
) -> List[int]:
    """
    MMR (Maximal Marginal Relevance) の簡易実装。
    - relevance: query と doc の類似度
    - redundancy: 既選択 doc との最大類似
    返り値: 選択した doc のインデックス（doc_vecs の行番号）
    """
    assert doc_vecs.ndim == 2
    sims = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()  # cosine (正規化済み想定)
    selected: List[int] = []
    candidates = list(range(doc_vecs.shape[0]))

    while candidates and len(selected) < top_k:
        if not selected:
            # 最初は純粋に関連度最大
            best = int(np.argmax(sims[candidates]))
            selected.append(candidates.pop(best))
            continue

        # 冗長性（既選択との最大類似）を引く
        red = np.max(doc_vecs[candidates] @ doc_vecs[selected].T, axis=1)
        score = lambda_ * sims[candidates] - (1 - lambda_) * red
        pick = int(np.argmax(score))
        selected.append(candidates.pop(pick))

    return selected


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def build_seed(
    query: str,
    picked_docs: List[Dict[str, Any]],
    constraints: str,
    max_sources: int = 6,
    abstract_chars: int = 0,
) -> Dict[str, Any]:
    """
    picked_docs: [{"title","abstract","fields":[...]}, ...] を想定
    - sources を fields のユニーク集合から抽出
    - problem は代表（1件目）の title を使用（必要に応じて変更可）
    - abstract は長すぎる場合はトリム
    """
    # sources = fields のユニーク集合（順序維持）
    src_od = OrderedDict()
    for d in picked_docs:
        for f in d.get("fields", []) or []:
            if f not in src_od:
                src_od[f] = True
    sources = list(src_od.keys())[:max_sources] or ["LLM"]

    context = []
    for d in picked_docs:
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()
        if abstract_chars > 0:
            abstract = truncate_text(abstract, abstract_chars)
        context.append(
            {
                "title": title,
                "abstract": abstract,
                "fields": d.get("fields", []) or [],
            }
        )

    seed = {
        "topic": query,
        "problem": picked_docs[0].get("title", query) if picked_docs else query,
        "sources": sources,
        "constraints": constraints,
        "context": context,
    }
    return seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="./data/queries.txt")
    ap.add_argument("--index", default="./rag/index.faiss")
    ap.add_argument("--meta", default="./rag/meta.jsonl")
    ap.add_argument("--out", default="./data/research_seeds.jsonl")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=5, help="MMR後の選択数（文脈ドキュメント数）")
    ap.add_argument("--mmr_lambda", type=float, default=0.6, help="MMRの関連性重み（0.0〜1.0）")
    ap.add_argument("--search_topn", type=int, default=50, help="MMR前に取る候補数")
    ap.add_argument("--abstract_chars", type=int, default=0, help="abstractの最大文字数（0で無制限）")
    ap.add_argument("--constraints", default="real-world viability", help="シードに入れる制約文字列")
    ap.add_argument("--max_sources", type=int, default=6, help="sourcesに入れる分野の最大数")
    ap.add_argument("--dedupe_titles", action="store_true", help="タイトル重複の除去（全シード横断）")
    args = ap.parse_args()

    # 読み込み
    print(f"[INFO] loading FAISS index: {args.index}")
    index = faiss.read_index(args.index)

    print(f"[INFO] loading meta: {args.meta}")
    metas = load_meta(args.meta)
    assert len(metas) == index.ntotal, "meta と index の件数が一致しません"

    print(f"[INFO] loading queries: {args.queries}")
    queries = load_queries(args.queries)

    print(f"[INFO] loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    # 出力先
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")

    # タイトル重複除去用セット
    seen_titles = set()

    for q in tqdm(queries, desc="queries"):
        # クエリ埋め込み
        q_vec = model.encode([q], normalize_embeddings=True)[0].astype("float32")

        # 候補検索
        D, I = index.search(q_vec.reshape(1, -1), args.search_topn)
        idxs = I[0].tolist()

        # 候補の埋め直し（タイトル+アブスト本文で埋めるのが無難）
        cand_docs: List[Dict[str, Any]] = []
        cand_vecs = []

        for idx in idxs:
            m = metas[idx]
            title = (m.get("title") or "").strip()
            abstract = (m.get("abstract") or "").strip()
            text = f"{title}\n{abstract}"
            vec = model.encode([text], normalize_embeddings=True)[0].astype("float32")
            cand_vecs.append(vec)
            cand_docs.append(
                {"title": title, "abstract": abstract, "fields": m.get("fields", []) or []}
            )

        cand_vecs = np.stack(cand_vecs, axis=0)

        # MMR で多様性確保
        pick = mmr(q_vec, cand_vecs, top_k=args.k, lambda_=args.mmr_lambda)
        picked_docs = [cand_docs[i] for i in pick]

        # タイトル重複除去（任意）
        if args.dedupe_titles:
            filtered = []
            for d in picked_docs:
                if d["title"] in seen_titles:
                    continue
                seen_titles.add(d["title"])
                filtered.append(d)
            if filtered:
                picked_docs = filtered

        # シードを構築して出力
        seed = build_seed(
            query=q,
            picked_docs=picked_docs,
            constraints=args.constraints,
            max_sources=args.max_sources,
            abstract_chars=args.abstract_chars,
        )
        fout.write(json.dumps(seed, ensure_ascii=False) + "\n")

    fout.close()
    print(f"[DONE] wrote seeds -> {args.out}")


if __name__ == "__main__":
    main()

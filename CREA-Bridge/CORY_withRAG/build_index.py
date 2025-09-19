#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_index.py
- 学術コーパス (papers.jsonl) からベクトル索引を作成して保存
- 出力: ./rag/index.faiss, ./rag/meta.jsonl
- 使い方:
    python build_index.py \
      --input ./data/papers.jsonl \
      --out_dir ./rag \
      --model sentence-transformers/all-MiniLM-L6-v2 \
      --batch_size 256
- 依存:
    pip install sentence-transformers faiss-cpu tqdm
"""

import os
import json
import argparse
import pathlib
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# pip: faiss-cpu (CPU版)、GPUの場合は faiss-gpu でも可
import faiss
from sentence_transformers import SentenceTransformer


def load_papers(path: str) -> List[Dict[str, Any]]:
    """papers.jsonl を読み込み、title+abstract がある行だけを返す"""
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            if not title or not abstract:
                continue
            # 余計な改行・空白を軽く整える
            obj["title"] = " ".join(title.split())
            obj["abstract"] = " ".join(abstract.split())
            # fields がなければ空配列
            if "fields" not in obj or not isinstance(obj["fields"], list):
                obj["fields"] = []
            items.append(obj)
    if not items:
        raise ValueError(f"No valid papers found in {path}")
    return items


def build_embeddings(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 256,
    normalize: bool = True,
) -> np.ndarray:
    """SentenceTransformers で埋め込みを作る（正規化オプションあり）"""
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    # float32 固定（FAISS の一般的想定）
    embs = embs.astype("float32")
    return embs


def save_faiss(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="./data/papers.jsonl", help="論文コーパス(JSONL)")
    ap.add_argument("--out_dir", default="./rag", help="出力ディレクトリ")
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="埋め込みモデル名（SentenceTransformers）",
    )
    ap.add_argument("--batch_size", type=int, default=256, help="埋め込み時のバッチサイズ")
    ap.add_argument(
        "--index_type",
        default="ip",
        choices=["ip", "l2"],
        help="FAISSインデックス種別（ip=内積: 正規化済みならコサイン相当 / l2=ユークリッド）",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="大規模時の分割作成用: 何分割中の何番目(0-based)。通常は0のままでOK",
    )
    ap.add_argument("--shards", type=int, default=1, help="分割数（通常は1）")
    args = ap.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] loading papers: {args.input}")
    papers = load_papers(args.input)

    # 分割（超大規模のときだけ使用）
    if args.shards > 1:
        n = len(papers)
        per = (n + args.shards - 1) // args.shards
        beg = args.shard * per
        end = min(n, beg + per)
        print(f"[INFO] sharding: {args.shard+1}/{args.shards} -> {beg}:{end} / {n}")
        papers = papers[beg:end]

    # 埋め込みテキスト（title + abstract）
    texts = [p["title"] + "\n" + p["abstract"] for p in papers]

    print(f"[INFO] loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print(f"[INFO] encoding {len(texts)} docs (batch_size={args.batch_size}) ...")
    embs = build_embeddings(model, texts, batch_size=args.batch_size, normalize=True)

    # 正規化済みベクトルに対しては inner product = cosine 相当
    if args.index_type == "ip":
        index = faiss.IndexFlatIP(embs.shape[1])
    else:  # l2
        index = faiss.IndexFlatL2(embs.shape[1])

    print("[INFO] adding vectors to index ...")
    index.add(embs)

    index_path = os.path.join(args.out_dir, "index.faiss")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")

    print(f"[INFO] saving index -> {index_path}")
    save_faiss(index, index_path)

    print(f"[INFO] saving metadata -> {meta_path}")
    with open(meta_path, "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[DONE] built index for {len(papers)} items")


if __name__ == "__main__":
    main()

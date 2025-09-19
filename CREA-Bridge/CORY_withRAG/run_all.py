#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all.py
- 1) build_index.py      : 論文コーパスから FAISS 索引を作成
- 2) search_to_seeds.py  : クエリ → RAG検索(MMR) → research_seeds.jsonl を生成
- 3) cory_withIRM_2_rag.py: RAG文脈を注入して CORY 学習 (Observer→Pioneer→IRM報酬)

使い方:
    python run_all.py \
      --papers ./data/papers.jsonl \
      --queries ./data/queries.txt \
      --out_dir ./rag \
      --seeds ./data/research_seeds.jsonl \
      --embed_model sentence-transformers/all-MiniLM-L6-v2 \
      --k 5 --mmr_lambda 0.6 \
      --model_name gpt2 \
      --irm_model_dir ./IRM/irm_iclr_model \
      --total_steps 1000

必要pip:
    pip install sentence-transformers faiss-cpu tqdm
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path


def run(cmd: list, cwd: str | None = None):
    print(f"[RUN] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] command failed: {e}", file=sys.stderr)
        sys.exit(1)


def ensure_exists(path: str, kind: str = "file"):
    p = Path(path)
    if kind == "file" and not p.is_file():
        print(f"[ERROR] {kind} not found: {path}", file=sys.stderr)
        sys.exit(1)
    if kind == "dir" and not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    # 入出力
    ap.add_argument("--papers", default="./data/papers.jsonl")
    ap.add_argument("--queries", default="./data/queries.txt")
    ap.add_argument("--out_dir", default="./rag")  # FAISS 出力
    ap.add_argument("--seeds", default="./data/research_seeds.jsonl")
    # 埋め込みモデル / RAG
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--search_topn", type=int, default=50)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mmr_lambda", type=float, default=0.6)
    ap.add_argument("--constraints", default="real-world viability")
    ap.add_argument("--abstract_chars", type=int, default=0)
    ap.add_argument("--max_sources", type=int, default=6)
    ap.add_argument("--dedupe_titles", action="store_true")
    # 学習設定（主要どころのみ転送）
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--irm_model_dir", default="./IRM/irm_iclr_model")
    ap.add_argument("--total_steps", type=int, default=1000)
    ap.add_argument("--batch_size_train", type=int, default=8)
    ap.add_argument("--swap_every", type=int, default=5)
    args, extra = ap.parse_known_args()

    # 0) 前提チェック
    ensure_exists(args.papers, "file")
    ensure_exists(args.queries, "file")
    ensure_exists(args.out_dir, "dir")
    ensure_exists(Path(args.seeds).parent.as_posix(), "dir")

    # 1) build_index.py
    run([
        sys.executable, "build_index.py",
        "--input", args.papers,
        "--out_dir", args.out_dir,
        "--model", args.embed_model,
        "--batch_size", str(args.batch_size),
    ])

    index_path = os.path.join(args.out_dir, "index.faiss")
    meta_path = os.path.join(args.out_dir, "meta.jsonl")
    ensure_exists(index_path, "file")
    ensure_exists(meta_path, "file")

    # 2) search_to_seeds.py
    cmd_seed = [
        sys.executable, "search_to_seeds.py",
        "--queries", args.queries,
        "--index", index_path,
        "--meta", meta_path,
        "--out", args.seeds,
        "--model", args.embed_model,
        "--k", str(args.k),
        "--mmr_lambda", str(args.mmr_lambda),
        "--search_topn", str(args.search_topn),
        "--constraints", args.constraints,
        "--max_sources", str(args.max_sources),
        "--abstract_chars", str(args.abstract_chars),
    ]
    if args.dedupe_titles:
        cmd_seed.append("--dedupe_titles")
    run(cmd_seed)
    ensure_exists(args.seeds, "file")

    # 3) cory_withIRM_2_rag.py
    run([
        sys.executable, "cory_withIRM_2_rag.py",
        "--model_name", args.model_name,
        "--irm_model_dir", args.irm_model_dir,
        "--seeds_path", args.seeds,
        "--total_steps", str(args.total_steps),
        "--batch_size", str(args.batch_size_train),
        "--swap_every", str(args.swap_every),
        # 追加のtyro引数を渡したい場合は run_all.py の引数末尾に付ければ extra に入り、そのまま渡せます
        *extra
    ])

    print("[DONE] all stages finished.")


if __name__ == "__main__":
    main()

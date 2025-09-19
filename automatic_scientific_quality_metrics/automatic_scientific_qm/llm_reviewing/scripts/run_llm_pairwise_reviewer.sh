#! /bin/bash 

set -e

# Run LLM pairwise comparison on ICLR
python run_llm_comparison.py --config ./configs/llm_comparison_review.yaml --dataset "iclr" --llm_provider $1

# Run LLM pairwise comparison on NeurIPS
python run_llm_comparison.py --config ./configs/llm_comparison_review.yaml --dataset "neurips" --llm_provider $1
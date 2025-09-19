#! /bin/bash

set -e 

mkdir -p model_store

# Run score prediction models for ICLR and NeurIPS
echo "Run score prediction models for ICLR and NeurIPS"
python run_score_models.py --config ./configs/rsp_review.yaml --dataset "iclr" --checkpoint ./model_store/iclr_score_prediction.ckpt
python run_score_models.py --config ./configs/rsp_review.yaml --dataset "neurips" --checkpoint ./model_store/neurips_score_prediction.ckpt

# Run pairwise comparison models for ICLR and NeurIPS
echo "Run pairwise comparison models for ICLR and NeurIPS"
python run_score_models.py --config ./configs/rsp_comparison_review.yaml --dataset "iclr" --checkpoint ./model_store/iclr_pairwise_comparison.ckpt
python run_score_models.py --config ./configs/rsp_comparison_review.yaml --dataset "neurips" --checkpoint ./model_store/neurips_pairwise_comparison.ckpt




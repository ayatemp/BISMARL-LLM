#! /bin/bash 

set -e 

echo "Running Sakana Reviewer on ICLR"
python run_sakana.py --config ./configs/sakana_review_iclr.yaml --llm_provider $1

echo "Running Sakana Reviewer on NeurIPS"
python run_sakana.py --config ./configs/sakana_review_neurips.yaml --llm_provider $1



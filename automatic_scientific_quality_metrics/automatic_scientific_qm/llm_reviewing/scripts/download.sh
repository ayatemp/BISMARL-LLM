#! /bin/bash

set -e

mkdir -p model_store

# Download models 
echo "Donwloading the score prediction models"

cd model_store

# Review Score Predictor ICLR Title + Abstract
if [ ! -f "iclr_score_prediction.ckpt" ]; then
    gdown 1n42AggcFJcoU-LSj0ME1wYCHbSrlLV-G
fi

# Review Score Predictor NeurIPS Title + Abstract
if [ ! -f "neurips_score_prediction.ckpt" ]; then
    gdown 1W1yFGXYBTrk4_-XesATYbsLq1jLzeF1H
fi

# Review Pairwise Comparison ICLR Title + Abstract
if [ ! -f "iclr_pairwise_comparison.ckpt" ]; then
    gdown 1IJAotMis3XBykWotdl4DItGO9WydzhS6
fi

# Review Pairwise Comparison NeurIPS Title + Abstract
if [ ! -f "neurips_pairwise_comparison.ckpt" ]; then
    gdown 1OBcyNVtp6653VTuPoBDx94z_GIU9LYF_ 
fi

cd ..


echo "Downloading the data"

mkdir -p data

cd data 

if [ ! -f "iclr_200_subset_data.pt" ]; then
    gdown 1c9jSkRSvPv4y6NaSgZdyAsajRyNIXiIr
fi

if [ ! -f "neurips_200_subset_data.pt" ]; then
    gdown 1RfqC0Wwh-J_kSlN-0MUbZxHrkdhX7n60
fi

cd ..
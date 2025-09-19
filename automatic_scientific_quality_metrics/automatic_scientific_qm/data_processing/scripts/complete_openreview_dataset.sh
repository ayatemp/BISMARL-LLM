#! /bin/bash

set -e 


# Download the section classifier model from Google Drive 
mkdir -p model_store
cd model_store
gdown 1HWIeDzumNDZFaOT-AvKl8uqSYppdzctA
cd ..

# Start grobid if session not already exists 
if tmux has-session -t run_grobid 2>/dev/null; then
    sleep 1
else
    tmux new-session -d -s run_grobid "bash ./scripts/run_grobid.sh"
    # Wait until GORBID started
    sleep 10
fi

# Run the script to complete the openreview dataset
python complete.py --config ./configs/complete_openreview.yaml --dataset $1

# Clean up the tmux session 
tmux kill-session -t run_grobid
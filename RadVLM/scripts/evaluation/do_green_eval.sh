#!/usr/bin/env bash

export DATA_DIR=/capstor/store/cscs/swissai/a135/RadVLM_project/data/

for json in "$@"; do
  torchrun --nproc_per_node=4 -m radvlm.evaluation.eval_green --results_path "$json"
done
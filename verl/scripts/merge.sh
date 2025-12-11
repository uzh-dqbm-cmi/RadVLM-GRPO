#!/bin/bash
# Usage: bash scripts/merge.sh /path/to/global_step_40 my_suffix

set -e

# Check arguments
if [ $# -ne 2 ]; then
  echo "Usage: $0 <base_path> <suffix>"
  exit 1
fi

BASE_PATH="${1%/}" # remove trailing slash if present
SUFFIX="$2"

# Define directories
ACTOR_DIR="${BASE_PATH}/actor"
TARGET_DIR="${BASE_PATH}/actor_hf_qwen_${SUFFIX}"

# Run merge
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_DIR" \
  --target_dir "$TARGET_DIR"

echo "Merge complete: $TARGET_DIR"
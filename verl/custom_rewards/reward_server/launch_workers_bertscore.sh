#!/usr/bin/env bash
set -euo pipefail

export WORKER_BASE_PORT=9400
export WORKER_NUM_GPUS=4
export WORKER_INSTANCES_PER_GPU=8
export WORKER_HOST="$(hostname -i | awk '{print $1}')"

export HF_HUB_OFFLINE=1

pip install -q fastapi uvicorn httpx

if ! pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
  nvidia-cuda-mps-control -d
fi

pids=()
port=$WORKER_BASE_PORT
ports=()

for gpu in $(seq 0 $((WORKER_NUM_GPUS-1))); do
  for inst in $(seq 0 $((WORKER_INSTANCES_PER_GPU-1))); do
    CUDA_VISIBLE_DEVICES=$gpu uvicorn worker_bertscore:app --host "$WORKER_HOST" --port $port --workers 1 &
    pids+=($!)
    ports+=($port)
    port=$((port+1))
    sleep 0.05
  done
done

echo "PORTS=$(IFS=,; echo "${ports[*]}")"
echo "COUNT=${#ports[@]}"
trap 'kill ${pids[@]} || true' EXIT
wait
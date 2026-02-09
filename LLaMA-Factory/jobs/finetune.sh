
# ─────────── cluster‐wide env ───────────────────
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0
export HF_HOME=$SCRATCH/huggingface_home
export WANDB_RUN_NAME=$SLURM_JOB_NAME
# ───────── repo setup & cd ─────────────────────────
export REPO_HOME=$SCRATCH/code/RadVLM-r1/LLaMA-Factory
cd $REPO_HOME

# ────────── launch distributed training ────────────
srun --label --export=ALL --environment=llama-factory bash -c '
  set -euo pipefail

  cd $REPO_HOME

  # Figure out this node’s rank by hostname lookup
  host_list=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
  rank=0
  for h in "${host_list[@]}"; do
    if [[ "$h" == "$(hostname)" ]]; then
      break
    fi
    rank=$((rank+1))
  done
  export NODE_RANK=$rank
  export NNODES=$SLURM_NNODES
  export FORCE_TORCHRUN=1

  echo "➤ Node $NODE_RANK / $NNODES starting training…"

  llamafactory-cli train examples/train_full/qwen2_5vl_full_sft.yaml
'


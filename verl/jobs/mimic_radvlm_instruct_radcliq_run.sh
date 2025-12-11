#!/usr/bin/env bash

export SLURM_CPUS_PER_TASK=288
export SLURM_GPUS=4
echo "${SLURM_PROCID}"

unset ROCR_VISIBLE_DEVICES
set -x
ENGINE=${1:-vllm}

export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC=1

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/
export TORCH_CUDA_ARCH_LIST="9.0a"
export WORK_DIR=/users/user/repos/RadVLM-GRPO/verl
export DATA_DIR=$SCRATCH/
cd $WORK_DIR

export SAVE_PATH=$SCRATCH/checkpoints/${SLURM_JOB_NAME}

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

WANDB_DIR=${SAVE_PATH}

# --- Assign nodes to roles
# master node is for ray master start
NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr "\n" " " | xargs)
NODES_ARR=($NODES)
MASTER_NODE=${NODES_ARR[0]}
REWARD_NODE=${NODES_ARR[1]}


export MASTER_NODE_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)

# ----- Reward Server exports

# get reward server ip
export REWARD_NODE_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$REWARD_NODE" hostname --ip-address)
export WORKER_BASE_PORT=9400
export WORKER_NUM_GPUS=4
export WORKER_INSTANCES_PER_GPU=8

echo "master node ip ${MASTER_NODE_IP}"
echo "reward node ip ${REWARD_NODE_IP}"

# ----- ray setup
export PORT=6553
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"

export WANDB_RUN_ID=${SLURM_JOB_NAME}
export WANDB_RUN_NAME=${SLURM_JOB_NAME}

uv pip install --system -q evaluate fastapi uvicorn httpx

# setup mpi style
# --- start ray master node
if [[ $SLURM_PROCID -eq 0 ]]; then
  echo "starting head"
  echo "head hostname: $(hostname)"
  echo "master node ip ${MASTER_NODE_IP}"
  ray start --head --node-ip-address=$MASTER_NODE_IP --port=$PORT --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS --block &

# --- start reward server node
elif [[ $SLURM_PROCID -eq 1 ]]; then
  echo "starting reward server"
  echo "reward hostname: $(hostname)"

  {

  cd ../RadEval
  uv pip install --system  .
  cd ../verl
  cd custom_rewards/reward_server

  export HF_HUB_OFFLINE=1


  if ! pgrep -f "nvidia-cuda-mps-control" >/dev/null; then
    nvidia-cuda-mps-control -d
  fi

  pids=()
  port=$WORKER_BASE_PORT
  ports=()

  mkdir -p "$SAVE_PATH/reward_server_logs"

  # export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:1024"

  for inst in $(seq 0 $((WORKER_INSTANCES_PER_GPU-1))); do
    for gpu in $(seq 0 $((WORKER_NUM_GPUS-1))); do
      log_file="$SAVE_PATH/reward_server_logs/uvicorn_gpu${gpu}_inst${inst}_port${port}.log"
      CUDA_VISIBLE_DEVICES=$gpu uvicorn worker_radcliq:app --host "$REWARD_NODE_IP" --port $port --workers 1 >"$log_file" 2>&1 &
      pids+=($!)
      ports+=($port)
      port=$((port+1))
      sleep 0.05
    done
  done

  echo "PORTS=$(IFS=,; echo "${ports[*]}")"
  echo "COUNT=${#ports[@]}"
  wait

  } > $SAVE_PATH/reward_server.log 2>&1

# ---- start ray worker nodes
else
  echo "starting non head"
  echo "worker hostname: $(hostname)"
  echo "master node ip ${MASTER_NODE_IP}"
  ray start --address=$RAY_ADDRESS --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS --block &
fi

sleep 10

ray status

# --- on master node start verl
if [[ $SLURM_PROCID -eq 0 ]]; then


export HF_HUB_OFFLINE=0
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/data/mimic_radvlm_instruct/train.parquet" \
    data.val_files="${DATA_DIR}/data/mimic_radvlm_instruct/val.parquet" \
    data.train_batch_size=512 \
    data.max_prompt_length=3072 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation="error" \
    data.image_key=images \
    actor_rollout_ref.model.path="/capstor/store/cscs/swissai/a135/RadVLM_project/models/qwen3VL_full_final" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="${WORK_DIR}/custom_rewards/radcliq_reward.py" \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive_pool \
    trainer.critic_warmup=0 \
    trainer.logger="[\"console\",\"wandb\"]" \
    trainer.project_name="verl_grpo_mimic_reportgen" \
    trainer.experiment_name=${SLURM_JOB_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=8 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.total_epochs=1 \

srun --overlap --ntasks=${SLURM_NNODES} ray stop --force

exit

else
wait
fi

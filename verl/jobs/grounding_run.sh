#export DATA_DIR=
#export MODEL_DIR=
#export WORK_DIR=

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


cd $WORK_DIR

export RUN_NAME=$SLURM_JOB_NAME

export WANDB_RUN_ID=$SLURM_JOB_NAME

export SAVE_PATH=/capstor/scratch/cscs/ndeperr/checkpoints/$SLURM_JOB_NAME

if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

WANDB_DIR=${SAVE_PATH}

NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr "\n" " " | xargs)
NODES_ARR=($NODES)
MASTER_NODE=${NODES_ARR[0]}
REWARD_NODE=${NODES_ARR[1]}

export MASTER_NODE_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$MASTER_NODE" hostname --ip-address)
export REWARD_NODE_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "$REWARD_NODE" hostname --ip-address)

echo "master node ip ${MASTER_NODE_IP}"
echo "reward node ip ${REWARD_NODE_IP}"

export PORT=6542
export RAY_ADDRESS="${MASTER_NODE_IP}:${PORT}"
echo "ray address ${RAY_ADDRESS}"


export WANDB_RESUME=allow      # or "must" to force-resume

pip install evaluate

if [[ $SLURM_PROCID -eq 0 ]]; then
  echo "starting head"
  echo "master node ip ${MASTER_NODE_IP}"
  ray start --head --node-ip-address=$MASTER_NODE_IP --port=$PORT --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS --block &

else
  echo "starting non head"
  echo "master node ip ${MASTER_NODE_IP}"
  ray start --address=$RAY_ADDRESS --num-cpus=$SLURM_CPUS_PER_TASK --num-gpus=$SLURM_GPUS --block &
fi

sleep 10

ray status

if [[ $SLURM_PROCID -eq 0 ]]; then

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=False \
    data.truncation="error" \
    data.image_key=images \
    actor_rollout_ref.model.path=${MODEL_PATH}  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=$WORK_DIR/custom_rewards/grounding_reward.py \
    custom_reward_function.name=compute_score \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger="[\"console\",\"wandb\"]" \
    trainer.project_name="verl_grpo_grounding" \
    trainer.experiment_name=$SLURM_JOB_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$SLURM_NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.total_epochs=10

srun --overlap --ntasks=$SLURM_NNODES ray stop --force

exit

else
wait
fi

#!/bin/bash
#SBATCH -A a135
#SBATCH --job-name=mimic_rg_radvlm_instruct_radcliq
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --output=job_outputs/%x_mimic_rg_radvlm_instruct_radcliq.out
#SBATCH --exclusive

WORK_DIR=/users/user/repos/RadVLM-GRPO/verl
LAUNCH=$WORK_DIR/jobs/mimic_radvlm_instruct_radcliq_run.sh

# use --environment here not in #SBATCH
srun --environment=vllm --gpus-per-task=4 bash "$LAUNCH"

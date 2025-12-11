#!/bin/bash
#SBATCH -A a135
#SBATCH --job-name=run_green
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --partition=normal
#SBATCH --environment=vllm
#SBATCH --time=12:00:00
#SBATCH --output=job_outputs/%x.out

set -euo pipefail

export DATA_DIR=/capstor/store/cscs/swissai/a135/RadVLM_project/data/

cd /users/user/repos/RadVLM-GRPO/repos_deps/GREEN

pip install -e . --no-deps

cd /users/user/repos/RadVLM-GRPO/RadVLM/

bash scripts/evaluation/do_green_eval.sh \
/path/to/actor_hf_qwen_radvlm_instruct_rl_radcliq_report_generation_output.json \
/path/to/actor_hf_qwen_radvlm_instruct_rl_bertscore_report_generation_output.json \
exit

export DATA_DIR=/capstor/store/cscs/swissai/a135/RadVLM_project/data/

pip install scikit_image==0.25.2 torchxrayvision==1.3.5 rouge_score==0.1.2 bert_score==0.3.13 f1chexbert==0.0.2 scikit-image ensemble_boxes
export TRANSFORMERS_VERBOSITY=info
export VLLM_NCCL_SO_PATH=/usr/lib/aarch64-linux-gnu/
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TORCH_CUDA_ARCH_LIST="9.0a"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC=1
python3 -m radvlm.evaluation.evaluate_instructions --task report_generation --model_name "/iopsstor/scratch/cscs/user/checkpoints/mimic_rg_radvlm_instruct_radcliq_attempt_1/global_step_300/actor_hf_qwen_radvlm_instruct_rl" --temp 0.0 --max_new_tokens 1024 --vllm

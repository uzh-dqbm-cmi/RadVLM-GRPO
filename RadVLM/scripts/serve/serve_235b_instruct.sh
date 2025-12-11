export TORCH_CUDA_ARCH_LIST="9.0a"
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
--tensor-parallel-size 4 \
--limit-mm-per-prompt.video 0 \
--async-scheduling \
--enable-expert-parallel \
--allowed-local-media-path /capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
--gpu-memory-utilization 0.90 \
--max_model_len 16384

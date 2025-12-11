export TORCH_CUDA_ARCH_LIST="9.0a"
vllm serve Qwen/Qwen3-VL-32B-Thinking \
--tensor-parallel-size 4 \
--limit-mm-per-prompt.video 0 \
--limit-mm-per-prompt.image 1 \
--async-scheduling \
--allowed-local-media-path /capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
--gpu-memory-utilization 0.9 \
--max_model_len 32768

#vllm serve Qwen/Qwen3-VL-32B-Instruct \
#--tensor-parallel-size 4 \
#--limit-mm-per-prompt.video 0 \
#--limit-mm-per-prompt.image 0 \
#--async-scheduling \
#--allowed-local-media-path /capstor/store/cscs/swissai/a135/RadVLM_project/data/ \
#--gpu-memory-utilization 0.45 \
#--max_model_len 32768 &

FROM docker.io/vllm/vllm-openai:v0.11.0

ARG MAX_JOBS=16
ENV MAX_JOBS=${MAX_JOBS}
ARG NVCC_THREADS=16
ENV NVCC_THREADS=${NVCC_THREADS}

ARG TORCH_CUDA_ARCH_LIST="9.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG VLLM_FA_CMAKE_GPU_ARCHES="90a-real"
ENV VLLM_FA_CMAKE_GPU_ARCHES=${VLLM_FA_CMAKE_GPU_ARCHES}
 
ENV MAKEFLAGS="-j16"
ENV CMAKE_BUILD_PARALLEL_LEVEL=16
ENV NINJA_BUILD_PARALLELISM=16
 
ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt update
RUN apt upgrade -y
RUN apt install -y --no-install-recommends \
    curl \
    gcc-12 g++-12 \
    git \
    libibverbs-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    zlib1g-dev \
    build-essential cmake ninja-build git ccache pkg-config \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/pip && cat > /etc/pip/constraints.txt <<'EOF'
numpy==1.26.4
qwen-vl-utils==0.0.14
xformers==0.0.33+f204359.d20251007
pyarrow>=19.0.0
EOF

# Make uv use it automatically
ENV UV_CONSTRAINT="/etc/pip/constraints.txt"

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -U build cmake ninja "packaging>=24"  pybind11 "setuptools>=77"  wheel

# Set compiler paths
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system numpy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-build-isolation git+https://github.com/facebookresearch/xformers.git@main#egg=xformers

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-build-isolation flash-attn

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system  qwen-vl-utils accelerate codetiming datasets dill hydra-core liger-kernel numpy pandas peft pyarrow pybind11 pylatexenc pre-commit "ray[default]" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata transformers wandb  uvicorn fastapi latex2sympy2_extended math_verify tensorboard hf_transfer modelscope bitsandbytes timm boto3 runai-model-streamer runai-model-streamer[s3] tensorizer  nltk  radgraph codetiming datasets dill hydra-core pandas peft  pybind11 pylatexenc torchdata wandb scikit-image==0.25.2 ensemble-boxes==1.0.9 torchxrayvision==1.3.5 pydicom==3.0.1 sentencepiece==0.2.0 faster-coco-eval==1.6.7 rouge-score==0.1.2 bert-score==0.3.13 radgraph==0.1.18 f1chexbert==0.0.2 torchmetrics==1.8.0 albumentations==2.0.8  sentence-transformers==5.0.0 numba==0.59.1 llvmlite==0.42.0 grpcio==1.62.1 protobuf==4.24.4 scikit_image==0.25.2 torchxrayvision==1.3.5 rouge_score==0.1.2 bert_score==0.3.13 f1chexbert==0.0.2

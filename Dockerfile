FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Hugging Face cache
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface

WORKDIR /app

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \
    gcc \
    g++ \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Python packaging tools
# ------------------------------------------------------------

RUN python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    packaging

# ------------------------------------------------------------
# Verify CUDA compiler exists
# ------------------------------------------------------------

RUN nvcc --version

# ------------------------------------------------------------
# Copy Python requirements
# ------------------------------------------------------------

COPY requirements.txt .

# ------------------------------------------------------------
# Install the exact PyTorch version used by RSCoVLM
# ------------------------------------------------------------

RUN pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ------------------------------------------------------------
# Install application dependencies
# ------------------------------------------------------------

RUN pip install -r requirements.txt

# ------------------------------------------------------------
# Download official RSCoVLM repository
# ------------------------------------------------------------

RUN git clone --depth 1 \
    https://github.com/VisionXLab/RSCoVLM.git \
    /opt/RSCoVLM

WORKDIR /opt/RSCoVLM

# ------------------------------------------------------------
# Install RSCoVLM
# ------------------------------------------------------------

RUN pip install -e .

# ------------------------------------------------------------
# Return to worker directory
# ------------------------------------------------------------

WORKDIR /app

COPY handler.py .

# ------------------------------------------------------------
# Runtime configuration
# ------------------------------------------------------------

ENV MODEL_ID=Qingyun/RSCoVLM-7B-2512
ENV MAX_NEW_TOKENS=512
ENV MIN_PIXELS=200704
ENV MAX_PIXELS=1003520

# ------------------------------------------------------------
# Final environment verification
# ------------------------------------------------------------

RUN python3 -c "\
import torch; \
print('PyTorch:', torch.__version__); \
print('PyTorch CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
import transformers; \
print('Transformers:', transformers.__version__); \
from transformers import Qwen2_5_VLForConditionalGeneration; \
print('Qwen2.5-VL import: OK') \
"

# ------------------------------------------------------------
# Start RunPod worker
# ------------------------------------------------------------

CMD ["python3", "handler.py"]

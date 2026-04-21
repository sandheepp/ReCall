# Base: CUDA 12.4 + cuDNN dev (required by vllm 0.8.4 and flash-attn)
# Platform pinned to amd64 — vast.ai GPU nodes are x86_64; building on Apple Silicon requires this
FROM --platform=linux/amd64 nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CONDA_AUTO_ACCEPT_TOS=yes

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential ca-certificates \
    libssl-dev libffi-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda (needed for faiss-gpu which can't be installed via pip)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels --channel defaults \
    && conda clean -afy

# Create the re-call conda environment with Python 3.10
RUN conda create -n re-call python=3.10 -y && conda clean -afy

# Install faiss-gpu via conda (pip-installed faiss is incompatible)
RUN conda run -n re-call conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y \
    && conda clean -afy

WORKDIR /workspace/ReCall

# Copy project source
COPY . .

# Install PyTorch first (matching CUDA 12.4) then project deps
RUN conda run -n re-call pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the project in editable mode (installs all deps from setup.py)
RUN conda run -n re-call pip install --no-cache-dir -e .

# Install flash-attn (must be built against the already-installed torch/CUDA)
RUN conda run -n re-call pip install --no-cache-dir flash-attn --no-build-isolation

# Make conda env the default Python for subsequent RUN/CMD
ENV PATH=${CONDA_DIR}/envs/re-call/bin:${PATH}

# Expose ports for sandbox and retriever services
EXPOSE 8000 8001

CMD ["bash"]

# Base image: CUDA 11.8 with cuDNN8, Ubuntu 20.04, Python 3.8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Avoid user prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools

# (Optional) Set a working directory
WORKDIR /app

# Copy your code into the container
COPY . /app

# Install PyTorch with CUDA and core Python packages
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face Transformers, Datasets, and other requirements
RUN pip install \
    transformers \
    diffusers \
    accelerate \
    opencv-python \
    pillow \
    requests \
    tqdm \
    fastapi \
    uvicorn \
    git+https://github.com/huggingface/peft.git

# (Optional) If you have a requirements.txt, install it last for caching
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Default command (adjust if you use FastAPI, Flask, etc.)
CMD ["python", "handler.py"]

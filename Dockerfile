# Use an official PyTorch image with CUDA, Python 3.11, and cuDNN, compatible with RunPod A5000
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /video-caption-endpoint

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install system dependencies if needed (add more if you use them)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Set environment variables if needed (RunPod recommends these for FastAPI endpoints)
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI app (handler.py should contain 'app' for FastAPI)
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]

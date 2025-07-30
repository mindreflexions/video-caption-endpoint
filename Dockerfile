FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

# Install OS dependencies for OpenCV and ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install numpy<2.0.0
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-u", "handler.py"]

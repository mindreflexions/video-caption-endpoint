FROM python:3.10-slim

# Install system dependencies (ffmpeg needed for video handling)
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start handler
CMD ["python", "handler.py"]

FROM python:3.10-slim

# Install system dependencies (ffmpeg, etc.)
RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the handler
CMD ["python", "handler.py"]

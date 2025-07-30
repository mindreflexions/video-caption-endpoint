# Use official PyTorch image with CUDA and Python 3.10+
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy all your app code into the image
COPY . .

# Expose port for FastAPI (adjust if needed)
EXPOSE 8000

# Run your FastAPI app (replace main:app if your entrypoint is different)
CMD ["uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8000"]

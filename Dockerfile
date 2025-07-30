# Use a lightweight Python base image
FROM python:3.10-slim

# Install required system packages (libGL needed for OpenCV)
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Define the entry point
CMD ["python", "handler.py"]

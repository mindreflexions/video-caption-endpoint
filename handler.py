# handler.py

import runpod
import requests
import tempfile
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os

# Load BLIP2 model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

def download_video(video_url):
    response = requests.get(video_url, stream=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    for chunk in response.iter_content(chunk_size=8192):
        temp_file.write(chunk)
    temp_file.close()
    return temp_file.name

def extract_frames(video_path, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            # Convert frame to PIL Image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frames.append(pil_image)
        frame_count += 1

    cap.release()
    return frames

def caption_frames(frames):
    captions = []
    for idx, image in enumerate(frames):
        inputs = processor(images=image, return_tensors="pt").to(model.device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append({"frame": idx, "caption": caption})
    return captions

def handler(event):
    video_url = event.get("input", {}).get("video_url")
    if not video_url:
        return {"error": "Missing 'video_url' in input."}

    video_path = download_video(video_url)
    frames = extract_frames(video_path, frame_interval=30)  # every 30th frame (~1 per second for 30fps)
    captions = caption_frames(frames)
    os.remove(video_path)

    return {"captions": captions}

runpod.serverless.start({"handler": handler})

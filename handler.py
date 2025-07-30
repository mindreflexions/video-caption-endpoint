import requests
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def download_video(video_url, output_path):
    r = requests.get(video_url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_frames(video_path, frame_interval=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames

def generate_captions(frames):
    captions = []
    for frame in frames:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        captions.append(caption)
    return captions

def handler(event):
    video_url = event["input"].get("video_url")
    output_path = "/tmp/video.mp4"
    
    download_video(video_url, output_path)
    frames = extract_frames(output_path, frame_interval=30)
    captions = generate_captions(frames)

    return {"captions": captions}

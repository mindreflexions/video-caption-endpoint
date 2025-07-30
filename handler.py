import runpod
import cv2
import os
import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)

def download_video(video_url, filename="input_video.mp4"):
    response = requests.get(video_url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def extract_frame_captions(video_path, frame_interval=15):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    captions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert frame to RGB and PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Process and generate caption
            inputs = processor(images=image, return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=20)
            caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            captions.append({
                "frame": frame_count,
                "caption": caption
            })

        frame_count += 1

    cap.release()
    return captions

def handler(event):
    try:
        video_url = event["input"]["video_url"]
        video_path = download_video(video_url)

        frame_captions = extract_frame_captions(video_path)

        return {
            "captions": frame_captions,
            "frame_count": len(frame_captions)
        }

    except Exception as e:
        return {"error": str(e)}

# Start the RunPod serverless handler
if __name__ == "__main__":
    print("Starting handler...")
    runpod.serverless.start({"handler": handler})

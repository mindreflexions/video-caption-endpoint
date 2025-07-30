import os
import cv2
from fastapi import FastAPI, File, UploadFile
import uvicorn
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load BLIP model and processor once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def extract_frames(video_path, frame_rate=1):
    """
    Extracts one frame per second (default) from the video.
    Returns a list of image file paths.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count, frames = 0, []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % (fps * frame_rate) == 0:
            frame_path = f"frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        count += 1
    cap.release()
    return frames

def caption_image(image_path):
    """
    Generates a detailed caption for a single image.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = "Describe this image in great detail for video scene regeneration."
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_length=75, num_beams=5)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def analyze_frames(video_path):
    """
    Extracts frames and generates detailed captions for each frame.
    """
    frames = extract_frames(video_path, frame_rate=1)  # 1 frame per second (adjust if needed)
    captions = []
    for idx, frame in enumerate(frames):
        try:
            caption = caption_image(frame)
            captions.append({"frame": idx + 1, "caption": caption})
        except Exception as e:
            captions.append({"frame": idx + 1, "caption": f"Error: {e}"})
        os.remove(frame)  # Clean up temp frame file
    return captions

app = FastAPI()

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    video_path = f"./{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())
    captions = analyze_frames(video_path)
    os.remove(video_path)  # Clean up uploaded file
    return {"captions": captions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

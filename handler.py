import runpod
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cv2
import os
import tempfile
import shutil

# Load BLIP model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def extract_frames(video_path, fps=1):
    """Extract frames at the specified FPS from a video file."""
    output_dir = tempfile.mkdtemp()
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        return [], f"Unable to open video file at {video_path}"
    frame_count = 0
    frames = []
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    interval = int(frame_rate / fps) if frame_rate and frame_rate > 0 else 1
    success, image = vidcap.read()
    while success:
        if frame_count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, image)
            frames.append(frame_path)
        success, image = vidcap.read()
        frame_count += 1
    vidcap.release()
    return frames, None

def generate_captions(frames):
    captions = []
    for frame_path in frames:
        try:
            img = Image.open(frame_path).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append({
                "frame": os.path.basename(frame_path),
                "caption": caption
            })
        except Exception as e:
            captions.append({
                "frame": os.path.basename(frame_path),
                "error": str(e)
            })
    return captions

def handler(event):
    input_data = event.get("input", {})
    video_path = input_data.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return {"error": f"video_path missing or not found: {video_path}"}
    frames, err = extract_frames(video_path)
    if err:
        return {"error": err}
    captions = generate_captions(frames)
    # Cleanup extracted frames
    if frames:
        try:
            shutil.rmtree(os.path.dirname(frames[0]))
        except Exception:
            pass
    return {"captions": captions}

runpod.serverless.start({"handler": handler})

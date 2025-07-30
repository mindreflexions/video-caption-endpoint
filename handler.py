from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import base64
import io
import torch

app = FastAPI()

# Load model and processor on startup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class FrameRequest(BaseModel):
    frame: str  # base64-encoded image

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/run")
def run_inference(request: FrameRequest):
    try:
        # Decode base64 to image
        image_bytes = base64.b64decode(request.frame)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Could not decode image: {str(e)}"}

    # Inference
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return {"caption": caption}

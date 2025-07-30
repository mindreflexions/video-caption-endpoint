from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def run_on_folder(folder_path):
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".jpg") or fname.endswith(".png"):
            path = os.path.join(folder_path, fname)
            caption = caption_image(path)
            results.append({ "file": fname, "caption": caption })
    return results

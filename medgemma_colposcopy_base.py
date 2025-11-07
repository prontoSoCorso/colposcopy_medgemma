from transformers import pipeline
from PIL import Image
import requests
import torch
from huggingface_hub import login

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    dtype=torch.bfloat16,
    device="cuda",
)

# Image attribution: Stillwaterising, CC0, via Wikimedia Commons
# import image from "/home/phd2/Scaricati/m1.jpg"
image = Image.open("/home/phd2/Scaricati/mx.jpg").convert("RGB")

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are an expert gynecologist."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this colposcopy image. Classify it as 0 (negative) or 1 (LSIL) or 2 (HSIL/cancer). Respond only with the number."},
            {"type": "image", "image": image}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])

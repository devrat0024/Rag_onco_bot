from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "ai4bharat/indictrans2-en-indic-1B"
print("⏳ Loading IndicTrans2 model...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✅ Model loaded successfully!")

@app.post("/translate")
async def translate_text(request: Request):
    data = await request.json()
    text = data.get("q", "")
    if not text.strip():
        return {"translatedText": ""}
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=5, max_length=256)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"translatedText": translated_text}

@app.get("/")
def home():
    return {"status": "IndicTrans2 Translation API running ✅"}

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from src.inference.inference import generate_pros_cons

# ---------------------------
# Paths / IDs
# ---------------------------
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
peft_model_id = "sdelowar2/mistral-7B-reviews-insight-lora"

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(title="Review Insights API", version="1.0")

# ---------------------------
# Load tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------
# Quantization config (4-bit)
# ---------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ---------------------------
# Load model (base + LoRA adapter)
# ---------------------------
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.eval()

# ---------------------------
# Request schema
# ---------------------------
class ReviewRequest(BaseModel):
    reviews: list[str]


# ---------------------------
# API Endpoint
# ---------------------------
@app.post("/generate")
async def generate(request: ReviewRequest):
    try:
        result = generate_pros_cons(tokenizer=tokenizer, model=model, reviews=request.reviews)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

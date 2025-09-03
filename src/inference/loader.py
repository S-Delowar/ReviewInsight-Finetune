import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from  src.utils.config_loader import load_config

base_model_id = load_config()["fine_tune"]["base_model"]
adapter_model_id = load_config()["huggingface"]["finetuned_model_repo"]

def load_model():
    """
    Load fine tuned model from huggingface.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, adapter_model_id)
    model.eval()

    return tokenizer, model

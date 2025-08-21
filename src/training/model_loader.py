import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model
from huggingface_hub import login

import os 
from dotenv import load_dotenv

from src.utils.config_loader import load_config
from src.training.peft_config import get_peft_config

load_dotenv()

cfg = load_config()["fine_tune"]

# Configure quantization
if cfg["base_model"]["load_in_4bit"] == True:
    quant_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_quant_type="nf4"
    )
else:
    quant_config = BitsAndBytesConfig(
      load_in_8bit=True,
      bnb_8bit_compute_dtype=torch.bfloat16
    )
  
  
# Login to Hugging Face Hub
login(token=os.getenv("HF_TOKEN"))


def load_tokenizer() -> AutoTokenizer:
    """
    Load and configure the tokenizer for the base model.

    Returns:
        AutoTokenizer: Hugging Face tokenizer with padding and EOS token setup.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_peft_model() -> AutoModelForCausalLM:
    """
    - Load the base causal language model with quantization,
    - Enable gradient checkpointing to reduce memory usage during fine-tuning, and 
    - Wrap it with PEFT.

    Returns:
        AutoModelForCausalLM: PEFT-wrapped model ready for fine-tuning.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["name"],
        quantization_config = quant_config,
        device_map = "auto"  
    )
    
    base_model.gradient_checkpointing_enable()

    peft_config = get_peft_config()
    peft_model = get_peft_model(base_model, peft_config)
    print(f"Memory footprint for the model: {peft_model.get_memory_footprint() / 1024**2:.2f} MB")
    return peft_model




if __name__ == "__main__":
    model = load_peft_model()
    print(model.get_memory_footprint())
    
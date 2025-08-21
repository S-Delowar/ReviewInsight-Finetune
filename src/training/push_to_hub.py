from transformers import AutoTokenizer
from peft import PeftModel

from src.utils.config_loader import load_config

# Hugging Face repo
hf_model_repo = load_config()["huggingface"]["finetuned_model_repo"]


# Path to local saved model & tokenizer
output_dir = "./artifacts/finetuned_model"

# Load saved artifacts
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = PeftModel.from_pretrained(output_dir)

# Push to Hugging Face Hub
tokenizer.push_to_hub(hf_model_repo, commit_message="Upload tokenizer")
model.push_to_hub(hf_model_repo, commit_message="Upload fine tuned LoRA adapters")

print(f"Tokenizer and adapters pushed successfully to HuggingFace Hub at repo: {hf_model_repo} .")

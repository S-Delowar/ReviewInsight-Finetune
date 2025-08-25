import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils.config_loader import load_config

config = load_config()
base_model_id = config["fine_tune"]["base_model"]["name"]
fine_tuned_model_id = config["huggingface"]["finetuned_model_repo"]

def main(local_saved_dir:str):
    """
    Push locally saved adapters + tokenizer to Hugging Face Hub.
    """
    
    # Load base model + adapters
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    peft_model = PeftModel.from_pretrained(base_model, local_saved_dir)
    
    # Push adapters
    print(f"Pushing adapters to hub")
    peft_model.push_to_hub(fine_tuned_model_id)
    
    # Push tokenizer
    tokenizer = AutoTokenizer.from_pretrained(local_saved_dir)
    tokenizer.push_to_hub(fine_tuned_model_id)
    
    print(f"Push completed successfully: {fine_tuned_model_id}")
    
    
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Push PEFT adapters and tokenizer to HuggingFace Hub")
    parser.add_argument("--local_saved_dir", type=str, required=True,
                        help="Local path to saved adapters")
    args = parser.parse_args()
    
    main(local_saved_dir=args.local_saved_dir)
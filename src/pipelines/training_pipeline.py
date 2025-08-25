import os
import argparse
from src.training.dataset_processing import load_process_dataset
from src.training.model_loader import load_tokenizer, load_peft_model
from src.training.trainer import get_trainer
from src.training.wandb_setup import init_wandb
from src.utils.config_loader import load_config

hf_model_repo_finetuned = load_config()["huggingface"]["finetuned_model_repo"]


def main(save_dir: str) -> None:
    """
    Train PEFT model and save adapters + tokenizer locally.
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb logging
    init_wandb()
    
    # Load tokenizer, dataset, and model
    tokenizer = load_tokenizer()
    dataset = load_process_dataset(tokenizer)
    model = load_peft_model()
    
    # Prepare trainer
    trainer = get_trainer(model, dataset)
    
    print("========Training Starts=======")
    trainer.train()
    
    print("========Training Finished!=======")

    # Save final artifacts manually
    tokenizer.save_pretrained(save_dir)
    trainer.model.save_pretrained(save_dir)
    print(f"Tokenizer and adapters saved successfully to {save_dir}")


if __name__=="__main__":  
    parser = argparse.ArgumentParser(description="Train PEFT model and save adapters")
    parser.add_argument("--save_dir", type=str, default="artifacts/finetuned_model")
    
    args = parser.parse_args()
    
    main(save_dir=args.save_dir)
    
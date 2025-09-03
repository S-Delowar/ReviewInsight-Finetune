import os
import argparse
from src.training.model_loader import load_tokenizer, load_peft_model
from src.training.wandb_setup import init_wandb
from src.utils.config_loader import load_config
from src.training.dataset_processing import load_process_dataset
from src.training.trainer import get_trainer

hf_model_repo_finetuned = load_config()["huggingface"]["finetuned_model_repo"]


def main(save_dir: str) -> None:
    """
    Train PEFT model and save adapters + tokenizer locally.
    """
    # Initialize wandb logging
    init_wandb()
    
    # Load tokenizer, dataset, and model
    tokenizer = load_tokenizer()
    dataset = load_process_dataset(tokenizer, subset_size=100)
    model = load_peft_model()
    
    # Prepare trainer
    trainer = get_trainer(model, tokenizer, dataset)
    
    print("========Training Starts=======")
    trainer.train()
    
    print("========Training Finished!=======")

    # Save final artifacts manually
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    trainer.model.save_pretrained(save_dir)
    print(f"Tokenizer and adapters saved successfully to {save_dir}")


if __name__=="__main__":  
    parser = argparse.ArgumentParser(description="Train PEFT model and save adapters")
    parser.add_argument("--save_dir", type=str, default="artifacts/finetuned_model")
    
    args = parser.parse_args()
    
    main(save_dir=args.save_dir)
    
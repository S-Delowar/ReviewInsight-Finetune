import os
from src.training.dataset_processing import load_process_dataset
from src.training.model_loader import load_tokenizer, load_peft_model
from src.training.trainer import get_trainer
from src.training.wandb_setup import init_wandb
from src.utils.config_loader import load_config

hf_model_repo_finetuned = load_config()["huggingface"]["finetuned_model_repo"]


def main(output_dir: str = "artifacts/finetuned_model") -> None:
    """
    Run the fine-tuning pipeline:
    - Initialize Weights & Biases logging
    - Load tokenizer, dataset, and PEFT model
    - Train the model with SFTTrainer
    - Save tokenizer and adapters locally to `output_dir`

    Args:
        output_dir (str, optional): Directory to save tokenizer and trained adapters.
                                    Defaults to './artifacts/finetuned_model'.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
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
    tokenizer.save_pretrained(output_dir)
    trainer.model.save_pretrained(output_dir)
    print(f"Tokenizer and adapters saved successfully to {output_dir}")


if __name__=="__main__":
    output_dir = os.getenv("OUTPUT_DIR", "artifacts/finetuned_model")
    main(output_dir=output_dir)
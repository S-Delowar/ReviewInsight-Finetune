import tokenize
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config

cfg = load_config()["fine_tune"]
dataset_repo = load_config()["huggingface"]["dataset_repo"]


def format_as_messages(example: dict) -> dict:
    """Convert a dataset example into a structured chat format."""
    
    review_text = "\n".join([f"- {r}" for r in example["input"]])
    user_prompt = f"Instruction: {example['instruction']}\nReviews:\n{review_text}"
    return {
        "messages": [
            {"role": "system", "content": "You are an assistant that extracts pros and cons from product reviews."},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["answer"]}
        ]
    }
    

def load_process_dataset(tokenizer, subset_size: int =10) -> dict:
    """
    Load dataset, split into train/val/test, format as chat messages, and tokenize.

    Args:
        tokenizer: Hugging Face tokenizer used for tokenization.
        subset_size (int, optional): Number of samples to use (for debugging). Defaults to 10.

    Returns:
        dict: Dictionary with train, validation, and test splits.
    """
    
    print("Dataset loading and preprocessing starts...")
    
    # Load full dataset
    dataset = load_dataset(dataset_repo, split="train")

    # Limit samples for debugging
    # dataset = dataset.select(range(min(subset_size, len(dataset))))
    
    # Ratios from config
    train_ratio, val_ratio, test_ratio = cfg["dataset"]["split_ratio"]

    # First split: train vs temp (val+test)
    train_test_split = dataset.train_test_split(test_size=(1 - train_ratio), seed=42)
    train_data = train_test_split["train"]
    temp = train_test_split["test"]

    # Second split: validation vs test
    val_test_split = temp.train_test_split(
        test_size=test_ratio / (val_ratio + test_ratio),
        seed=42
    )
    val_data = val_test_split["train"]
    test_data = val_test_split["test"]

    dataset_dict = {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
    }
    
    # Apply formatting
    for key in dataset_dict:
        dataset_dict[key] = dataset_dict[key].map(format_as_messages)
        
        # tokenizing
        def tokenize(example):
            print("Tokenizing is running")
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt = False 
            )
            return tokenizer(
                text,
                truncation=True,
                max_length = cfg["base_model"]["max_length"],
                padding = "max_length"
            )
            
        dataset_dict[key] = dataset_dict[key].map(tokenize, remove_columns=dataset_dict[key].column_names)
        
        print("Dataset processing done !")
    
    return dataset_dict
    
    


if __name__=="__main__":

    from src.training.model_loader import load_tokenizer
    
    tokenizer = load_tokenizer()
    dataset = load_process_dataset(tokenizer)
    
    print(dataset)
    print("===========================\n==============\n================\n")
    print(dataset["train"][1])
import json
import os
from datasets import Dataset

def save_to_jsonl(df, save_path: str) -> None:
    """
    Save dataset to JSONL format.
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save as JSONL format
    df.to_json(save_path, orient="records", force_ascii=False, lines=True)
    print(f"Saved {len(df)} records locally at {save_path}")


def push_to_huggingface(df, repo_id: str, private: bool = True) -> None:
    """
    Push dataset to Hugging Face Hub.
    """
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id, private=private)
    print(f"Dataset pushed to HuggingFace Hub at: {repo_id}")

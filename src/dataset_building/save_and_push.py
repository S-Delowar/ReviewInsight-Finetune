import json
import os
from datasets import Dataset

def save_to_jsonl(df, save_path: str):
    """
    Save dataset to JSONL format.
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
  
    with open(save_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    print(f"Saved dataset locally at {save_path}")


def push_to_huggingface(df, repo_id: str, private: bool = True):
    """
    Push dataset to Hugging Face Hub.
    """
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(repo_id, private=private)
    print(f"Dataset pushed to HuggingFace Hub at: {repo_id}")

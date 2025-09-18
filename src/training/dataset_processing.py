from datasets import load_dataset, DatasetDict
from src.utils.config_loader import load_config

cfg_root = load_config()
cfg = cfg_root["fine_tune"]
dataset_repo = cfg_root["huggingface"]["dataset_repo"]

SYSTEM_PROMPT = "You are an assistant that extracts pros and cons from product reviews. Return only valid JSON with keys pros and cons."

def _reviews_to_bullets(reviews):
    # Deterministic, readable presentation for the user turn
    return "\n".join([f"- {r.strip()}" for r in reviews if isinstance(r, str) and r.strip()])

def _build_messages(example: dict) -> dict:
    """
    Keep raw `answer` for evaluation and build chat `messages` for SFT.
    Do NOT tokenize here.
    """
    instruction = example.get("instruction", "").strip()
    reviews = example.get("input", [])
    answer = example.get("answer", "").strip()

    user_content = f"Instruction: {instruction}\nReviews:\n{_reviews_to_bullets(reviews)}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},  # gold JSON (text)
        ],
        "answer": answer,  # keep for metrics
    }

def load_process_dataset(tokenizer, subset_size: int | None = None) -> DatasetDict:
    """
    Load dataset from HF Hub, map to chat messages + keep raw gold 'answer'.
    """
    print("Dataset loading and preprocessing starts...")

    # Load the full split (assumed 'train' split holds all data)
    data = load_dataset(dataset_repo, split="train")

    # Optional subsetting for quick experiments
    if subset_size is not None:
        data = data.select(range(min(subset_size, len(data))))

    # Map to messages + answer (no tokenization)
    data = data.map(_build_messages, remove_columns=[c for c in data.column_names if c not in {"instruction", "input", "answer"}])

    # Split into train/val/test with fixed ratios
    train_ratio, val_ratio, test_ratio = cfg["dataset"]["split_ratio"]
    tmp_ratio = 1.0 - train_ratio

    split_1 = data.train_test_split(test_size=tmp_ratio, seed=42)
    train_data = split_1["train"]
    tmp = split_1["test"]

    val_size = test_ratio / (val_ratio + test_ratio)
    split_2 = tmp.train_test_split(test_size=val_size, seed=42)
    val_data = split_2["train"]
    test_data = split_2["test"]

    dataset = DatasetDict(
        train=train_data,
        validation=val_data,
        test=test_data,
    )

    print("Dataset processing done!")
    return dataset

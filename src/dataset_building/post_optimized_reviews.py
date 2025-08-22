"""
Optimize product reviews in a dataset by summarizing them with a transformer model.

Goal:
- Shorten long reviews into concise summaries.
- Reduce token count for efficient fine-tuning.

Process:
1. Load dataset from Hugging Face.
2. Summarize reviews in batches (GPU/CPU).
3. Save optimized dataset locally as JSON.
"""


import os
import gc
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline


def get_device():
    """Return device index for GPU or CPU"""
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("Using CPU")
    return -1


def build_summarizer(model_name, batch_size, device):
    """Load summarization pipeline"""
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=device,
        batch_size=batch_size,
        torch_dtype=torch.float16 if device >= 0 else None,
        clean_up_tokenization_spaces=True,
    )


def batch_summarize_reviews(summarizer, reviews, max_length, min_length, batch_size):
    """Summarize a list of reviews in batches with fallback on errors"""
    if not reviews:
        return []

    summarized_reviews = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batch_texts = [str(r).strip() if r else "" for r in batch]

        valid_texts = [(j, t) for j, t in enumerate(batch_texts) if t]
        empty_indices = [j for j, t in enumerate(batch_texts) if not t]

        try:
            results = summarizer(
                [t for _, t in valid_texts],
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            ) if valid_texts else []

            batch_summaries = [""] * len(batch)
            for j in empty_indices:
                batch_summaries[j] = ""
            for (orig_idx, _), res in zip(valid_texts, results):
                batch_summaries[orig_idx] = res["summary_text"].strip()

        except Exception as e:
            print(f"Batch error: {e}. Falling back to single reviews.")
            batch_summaries = []
            for r in batch:
                try:
                    if r and str(r).strip():
                        res = summarizer(
                            str(r).strip(),
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False,
                            truncation=True,
                        )
                        batch_summaries.append(res[0]["summary_text"].strip())
                    else:
                        batch_summaries.append("")
                except Exception as e:
                    print(f"Review error: {e}")
                    batch_summaries.append(" ".join(str(r).split()[:15]) if r else "")

        summarized_reviews.extend(batch_summaries)
    return summarized_reviews


def load_and_process_dataset(model_name, dataset_path, dataset_split,
                             max_length, min_length, batch_size):
    dataset = load_dataset(dataset_path, split=dataset_split)
    print(f"Loaded {len(dataset)} items")

    device = get_device()
    summarizer = build_summarizer(model_name, batch_size, device)

    processed_data = []
    total_reviews = sum(len(item["input"]) for item in dataset)
    print(f"Processing {len(dataset)} items with {total_reviews} total reviews...")

    for idx, item in enumerate(tqdm(dataset, desc="Processing items")):
        summarized_reviews = batch_summarize_reviews(
            summarizer, item["input"], max_length, min_length, batch_size
        )
        processed_data.append({
            "instruction": item["instruction"],
            "input": summarized_reviews,
            "answer": item["answer"]
        })

        if idx % 50 == 0 and idx > 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"✅ Progress: {idx+1}/{len(dataset)} items processed. Memory cleared.")

    return processed_data



if __name__ == "__main__":
    
    OUTPUT_FILE = "optimized_product_reviews_insight.json"
    MODEL_NAME = "facebook/bart-large-cnn"
    HF_DATASET_PATH = "sdelowar2/product-reviews-pros-cons"
    DATASET_SPLIT = "train"

    MAX_LENGTH = 25
    MIN_LENGTH = 5
    BATCH_SIZE = 4
    # ---------------------------------------- #

    data = load_and_process_dataset(
        model_name=MODEL_NAME,
        dataset_path=HF_DATASET_PATH,
        dataset_split=DATASET_SPLIT,
        max_length=MAX_LENGTH,
        min_length=MIN_LENGTH,
        batch_size=BATCH_SIZE,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"🎉 Completed! Saved {len(data)} items to {OUTPUT_FILE}")

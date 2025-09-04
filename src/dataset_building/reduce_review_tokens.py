"""
Summarize reviews from raw dataset to reduce token length.
"""
import argparse
import torch
import pandas as pd
from transformers import pipeline
from datasets import Dataset


def get_device():
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 0
    print("Using CPU")
    return -1


def build_summarizer(model_name, batch_size, device):
    return pipeline(
        "summarization",
        model=model_name,
        tokenizer=model_name,
        device=device,
        batch_size=batch_size,
        torch_dtype=torch.float16 if device >= 0 else None,
        clean_up_tokenization_spaces=True,
    )


def summarize_dataset(parquet_path, output_path, model_name, max_length, min_length, batch_size, start_index, end_index):
    df = pd.read_parquet(parquet_path)

    start_index = int(start_index)
    end_index = int(end_index)
    df = df[start_index:end_index]

    print(f"Summarizing Reviews from index:{start_index} to index:{end_index} in raw dataset")

    # Flatten all reviews across all rows into a single list with metadata
    flat_reviews = []
    review_indices = []

    for idx, row in df.iterrows():
        reviews_list = row["reviews"].tolist() if hasattr(row["reviews"], "tolist") else list(row["reviews"])
        for i, review in enumerate(reviews_list):
            flat_reviews.append(str(review).strip() if review else "")
            review_indices.append((idx, i))

    print(f"Total reviews to summarize: {len(flat_reviews)}")

    device = get_device()
    summarizer = build_summarizer(model_name, batch_size, device)

    # Convert to Dataset for efficient pipeline usage
    dataset = Dataset.from_dict({"text": flat_reviews})

    # Run summarization on the whole dataset
    summaries = summarizer(
        dataset["text"],
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
        batch_size=batch_size,
        clean_up_tokenization_spaces=True
    )

    # Reconstruct the structured output
    reconstructed_reviews = {}
    for (idx, i), summary in zip(review_indices, summaries):
        if idx not in reconstructed_reviews:
            reconstructed_reviews[idx] = []
        reconstructed_reviews[idx].append(summary["summary_text"].strip())

    # Build final DataFrame
    summarized_rows = [{"reviews": reconstructed_reviews.get(i, [])} for i in range(start_index, end_index)]
    summarized_df = pd.DataFrame(summarized_rows)
    summarized_df.to_parquet(output_path, index=False)
    print(f"✅ Saved summarized dataset to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize each reviews")
    parser.add_argument("--raw_data_path", type=str)
    parser.add_argument("--summarized_review_data_path", type=str)
    parser.add_argument("--start_index", type=str)
    parser.add_argument("--end_index", type=str)
    
    args = parser.parse_args()
    
    MODEL_NAME = "facebook/bart-large-cnn"

    MAX_LENGTH = 25
    MIN_LENGTH = 5
    BATCH_SIZE = 4

    summarize_dataset(
        parquet_path=args.raw_data_path, 
        output_path=args.summarized_review_data_path, 
        model_name=MODEL_NAME, 
        max_length=MAX_LENGTH, 
        min_length=MIN_LENGTH, 
        batch_size=BATCH_SIZE, 
        start_index=args.start_index,
        end_index=args.end_index
    )
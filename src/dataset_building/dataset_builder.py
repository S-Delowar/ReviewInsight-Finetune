"""
Build instruction dataset from summarized reviews (parallelized).
"""
import argparse
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import ast

from src.dataset_building.llm_utils import generate_pros_cons
from src.dataset_building.save_and_push import push_df_to_huggingface, save_to_jsonl
from src.utils.config_loader import load_config


HF_DATASET_REPO = load_config()["huggingface"]["dataset_repo"]

def process_row(row):
    instruction = "Generate pros and cons from the following product reviews."
    
    # Ensure reviews is a list
    reviews_list = row["reviews"]
    if hasattr(reviews_list, "tolist"):
        reviews_list = reviews_list.tolist()
    elif isinstance(reviews_list, str):
        try:
            reviews_list = ast.literal_eval(reviews_list)
        except Exception:
            reviews_list = [reviews_list]
    else:
        reviews_list = list(reviews_list)

    # Generate pros/cons
    try:
        answer = generate_pros_cons(reviews_list)
    except Exception as e:
        print(f"Error generating pros/cons: {e}")
        answer = {"pros": [], "cons": []}

    return {
        "instruction": instruction,
        "input": reviews_list,
        "answer": json.dumps(answer, ensure_ascii=False)
    }


def build_instruction_answer_dataset(parquet_path: str, sample_size: int = None, max_workers: int) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if sample_size:
        df = df.head(sample_size)

    records = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating pros/cons"):
            records.append(f.result())

    return pd.DataFrame(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build instruction dataset with pros/cons")
    parser.add_argument("--reviews_file_path", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON")
    parser.add_argument("--sample_size", type=int, help="Total samples")
    parser.add_argument("--max_workers", type=int, default=10, help="Number of parallel workers")

    args = parser.parse_args()

    print(f"Loading data from {args.reviews_file_path}")
    df_out = build_instruction_answer_dataset(parquet_path=args.reviews_file_path, max_workers=args.max_workers, sample_size=args.sample_size)

    print(f"Generated pros/cons dataset")
    
    # Save locally
    save_to_jsonl(df_out, args.output_file)
    
    # Push to HuggingFace Hub
    push_df_to_huggingface(df=df_out, repo_id=HF_DATASET_REPO, private=False)
    
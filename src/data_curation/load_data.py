import os 
import json 
import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """ 
    Load raw Amazon Review data from json file
    """
    print(f"Loading raw data from {file_path}...")
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    df = pd.DataFrame(data)
    
    # Keep only necessary columns and rename
    df = df[["asin", "reviewText", "summary"]]
    df.rename(columns={"asin": "product_id", "reviewText": "review_body", "summary":"review_summary"}, inplace=True)
        
    print(f"Loaded {len(df)} records with {len(df.columns)} columns.")
    return df


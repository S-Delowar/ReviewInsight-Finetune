import json
import pandas as pd
from src.dataset_building.llm_utils import generate_pros_cons

def build_instruction_answer_dataset(df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
    """
    Build dataset with instruction, input (reviews), and answer (pros/cons).
    """
    records = []
    # Take first N rows
    df_iter = df if sample_size is None else df.head(sample_size)
    
    print(f"Generating reviews pros and cons for {sample_size} products")

    for _, row in df_iter.iterrows():
        instruction = "Generate pros and cons from the following product reviews."
        reviews_list = row["reviews"].tolist() if hasattr(row["reviews"], "tolist") else list(row["reviews"])

        answer = generate_pros_cons(reviews_list)
        
        records.append({
            "instruction": instruction,
            "input": reviews_list,
            "answer": json.dumps(answer, ensure_ascii=False)
        })
        
    print(f"Pros-Cons dataset generated successfully.")

    return pd.DataFrame(records)

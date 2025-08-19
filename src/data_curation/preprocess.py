import pandas as pd

def preprocess_reviews(df: pd.DataFrame, review_col: str = "review_body") -> pd.DataFrame:
    """
    Preprocess Amazon review dataset.

    Steps:
    - Strip whitespaces
    - Drop null rows
    - Count word length
    - Keep reviews with 10–500 words
    - Truncate reviews longer than 200 words (keep first 200)
    - Return cleaned DataFrame
    """
    
    # Drop null reviews
    df = df.dropna(subset=[review_col]).copy()
    
    # Remove leading/trailing whitespaces
    df[review_col] = df[review_col].str.strip()
    
    # Count words
    df["word_count"] = df[review_col].apply(lambda x: len(str(x).split()))
    
    # Keep reviews with word count between 10 and 500
    df = df[(df["word_count"] >= 10) & (df["word_count"] <= 500)].copy()
    
    # Truncate reviews longer than 200 words
    def truncate_review(text):
        words = text.split()
        if len(words) > 200:
            return " ".join(words[:200])
        return text
    
    df[review_col] = df[review_col].apply(truncate_review)
    
    # Recompute word_count after truncation
    df["word_count"] = df[review_col].apply(lambda x: len(str(x).split()))
    
    return df.reset_index(drop=True)

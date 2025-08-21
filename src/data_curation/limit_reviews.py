import os
import random
import pandas as pd

def limit_reviews_per_product(df, save_path_preprocessed: str, min_reviews=5, max_reviews=15, review_col="reviews") -> pd.DataFrame:
    """
    Keep only products with a minimum number of reviews and limit maximum reviews per product.
    
    Args:
        df: DataFrame with grouped reviews
        min_reviews: minimum number of reviews to keep a product
        max_reviews: maximum number of reviews to keep per product
        review_col: column containing list of reviews
    
    Returns:
        DataFrame filtered by min_reviews and capped at max_reviews
    """
    df = df.copy()
    
    def cap_list(reviews):
        if len(reviews) > max_reviews:
            return random.sample(reviews, max_reviews) 
        return reviews
    
    # Apply cap
    df[review_col] = df[review_col].apply(cap_list)
    
    # Filter by minimum reviews
    df = df[df[review_col].apply(len) >= min_reviews]
    
    # Update review counts
    df["reviews_count"] = df[review_col].apply(len)
    
    if save_path_preprocessed:
        # Ensure parent directories exist
        os.makedirs(os.path.dirname(save_path_preprocessed), exist_ok=True)
        # Save to parquet
        df.to_parquet(save_path_preprocessed, index=False)
        print(f"Preprocessed final data saved to {save_path_preprocessed}")
        
    return df

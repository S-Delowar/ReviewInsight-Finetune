import pandas as pd

def group_reviews_by_product(df: pd.DataFrame, 
                             product_col: str = "product_id", 
                             review_col: str = "review_body") -> pd.DataFrame:
    """
    Group reviews into a list for the same product_id, 
    and add a column for number of reviews.

    Args:
        df: Preprocessed DataFrame containing reviews
        product_col: Column name for product id
        review_col: Column name for review text

    Returns:
        DataFrame with:
            - product_id
            - reviews (list of reviews)
            - reviews_count (# of reviews per product)
    """
    grouped_df = (
        df.groupby(product_col)[review_col]
        .apply(list)
        .reset_index()
        .rename(columns={review_col: "reviews"})
    )
    
    # Add review count column
    grouped_df["reviews_count"] = grouped_df["reviews"].apply(len)
    
    return grouped_df

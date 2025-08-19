import os
from src.data_curation.group_reviews import group_reviews_by_product
from src.data_curation.limit_reviews import limit_reviews_per_product
from src.data_curation.load_data import load_raw_data
from src.data_curation.preprocess import preprocess_reviews
from src.utils.config_loader import load_config


config = load_config("configs/config.yml")
raw_path = config["paths"]["raw_data"]
save_path_preprocessed = config["paths"]["preprocessed_data"]

def main():
    try:
        # Load raw data
        df = load_raw_data(raw_path)
        
        # Preprocess the reviews
        cleaned_df = preprocess_reviews(df)
        
        # Group reviews by product id
        product_reviews = group_reviews_by_product(cleaned_df)
        
        # Limit total number of reviews for each product
        final_product_reviews_df =limit_reviews_per_product(df=product_reviews, 
                                                            save_path_preprocessed=save_path_preprocessed, 
                                                            max_reviews=10,
                                                            min_reviews=5)
    except Exception as e:
        raise e
    
    
# ========================
if __name__ == "__main__":
    main()
import pandas as pd
from src.dataset_building.dataset_builder import build_instruction_answer_dataset
from src.dataset_building.save_and_push import save_to_jsonl, push_to_huggingface
from src.utils.config_loader import load_config


config=load_config()

curated_data_path = config["paths"]["preprocessed_data"]
generated_data_path = config["paths"]["generated_data"]
huggingface_repo = config["huggingface"]["dataset_repo"]


def main():
    try:
        # Load dataset
        df = pd.read_parquet(curated_data_path)

        # Build instruction-answer dataset
        instruction_answers_df = build_instruction_answer_dataset(df, sample_size=3000)

        # Save locally
        save_to_jsonl(instruction_answers_df, generated_data_path)

        # Push to HuggingFace Hub
        push_to_huggingface(df=instruction_answers_df, repo_id=huggingface_repo, private=False)
    except Exception as e:
        raise e


# ========================
if __name__ == "__main__":
    main()

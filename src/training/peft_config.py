from peft import LoraConfig
from src.utils.config_loader import load_config

peft_config = load_config()["fine_tune"]["peft"]


def get_peft_config() -> LoraConfig:
    """
    Build and return a LoRA configuration from project settings.

    Returns:
        LoraConfig: Configuration object for parameter-efficient fine-tuning.
    """
    return LoraConfig(
        r=peft_config["r"],
        lora_alpha=peft_config["lora_alpha"],
        target_modules=peft_config["target_modules"],
        lora_dropout=peft_config["lora_dropout"],
        bias=peft_config["bias"],
        task_type=peft_config["task_type"]
    )



if __name__ == "__main__":
    print(peft_config)
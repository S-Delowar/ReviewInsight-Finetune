import torch
from transformers import TrainingArguments, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import DatasetDict

from src.utils.config_loader import load_config


def get_trainer(model: AutoModelForCausalLM, dataset: DatasetDict) -> SFTTrainer:
    """
    Initialize and return an SFTTrainer with configuration from project settings.

    Args:
        model (PreTrainedModel): Hugging Face model to fine-tune.
        dataset (DatasetDict): Dataset containing 'train' and 'validation' splits.

    Returns:
        SFTTrainer: Trainer instance configured for supervised fine-tuning.
    """
    
    trainer_cfg = load_config()["fine_tune"]["training"]

    # Check if bfloat16 is supported
    bf16_supported = torch.cuda.is_bf16_supported()
    
    training_args = TrainingArguments(
        output_dir= trainer_cfg["output_dir"],
        num_train_epochs= int(trainer_cfg["num_train_epochs"]),
        learning_rate= float(trainer_cfg["learning_rate"]),
        lr_scheduler_type= trainer_cfg["lr_scheduler_type"],
        per_device_train_batch_size= int(trainer_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps= int(trainer_cfg["gradient_accumulation_steps"]),
        logging_steps= int(trainer_cfg["logging_steps"]),
        optim= trainer_cfg["optim"],
        weight_decay= float(trainer_cfg["weight_decay"]),
        warmup_steps= int(trainer_cfg["warmup_steps"]),
        seed= int(trainer_cfg["seed"]),
        save_strategy= trainer_cfg["save_strategy"],
        fp16= not bf16_supported,
        bf16= bf16_supported,
        report_to= "wandb"
    )    
    
    trainer = SFTTrainer(
        model= model,
        train_dataset= dataset["train"],
        eval_dataset= dataset["validation"],
        args= training_args,
    )
    
    print("Trainer initiated")
    
    return trainer
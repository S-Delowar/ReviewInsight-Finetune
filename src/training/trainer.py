import datetime
import packaging
import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from src.evaluation.compute_metrics import compute_metrics
from src.utils.config_loader import load_config

run_name = f"finetune-{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"

def get_trainer(model, tokenizer, dataset):
    """
    Build an SFTTrainer.
    """
    trainer_cfg = load_config()["fine_tune"]["training"]

    bf16_supported = torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=trainer_cfg["output_dir"],
        run_name=run_name,
        num_train_epochs=int(trainer_cfg["num_train_epochs"]),
        learning_rate=float(trainer_cfg["learning_rate"]),
        lr_scheduler_type=trainer_cfg["lr_scheduler_type"],
        per_device_train_batch_size=int(trainer_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(trainer_cfg["gradient_accumulation_steps"]),
        per_device_eval_batch_size=int(trainer_cfg["per_device_eval_batch_size"]),
        logging_steps=int(trainer_cfg["logging_steps"]),
        optim=trainer_cfg["optim"],
        weight_decay=float(trainer_cfg["weight_decay"]),
        warmup_steps=int(trainer_cfg["warmup_steps"]),
        seed=int(trainer_cfg["seed"]),
        evaluation_strategy=trainer_cfg["eval_strategy"],   
        save_strategy="epoch",        
        fp16=not bf16_supported,
        bf16=bf16_supported,
        report_to="wandb",
        load_best_model_at_end=False,
    )

    # Formatting function that uses the tokenizer's chat template
    def formatting_func(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return text

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        formatting_func=formatting_func,
        compute_metrics=compute_metrics,
    )

    print("Trainer initiated")
    return trainer

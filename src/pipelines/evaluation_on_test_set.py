import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.training.model_loader import get_quant_config
from src.training.wandb_setup import init_wandb
from src.training.dataset_processing import load_process_dataset
from src.evaluation.llm_judge import run_llm_judge
from src.utils.config_loader import load_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = load_config()["fine_tune"]
fine_tuned_model_id = load_config()["huggingface"]["finetuned_model_repo"]


def main(model_id: str = fine_tuned_model_id, subset_size_qualitative:int=5, device= DEVICE):
    # Init W&B
    init_wandb(
        run_name="evaluate_on_test_data",
        tags=["test-eval", "llm-judge"],
        notes=f"Evaluation on test set using checkpoint: {model_id}"
    )

    # Load tokenizer & PEFT model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"]["name"],
        quantization_config = get_quant_config(),
        device_map = "auto"  
    )
    model = PeftModel.from_pretrained(base_model, model_id)
    model.eval()
    model.to(device) 
    
    print("Loaded PEFT Model and Tokenizer")
    
    # Load dataset
    dataset = load_process_dataset(tokenizer=tokenizer)
    test_data = dataset["test"]
    
    print("Loaded Test dataset")

    # LLM-as-judge
    print(f"Judgement starts with GPT-4o-mini")
    table = run_llm_judge(model, tokenizer, test_data, DEVICE, subset_size=subset_size_qualitative)
    wandb.log({"qualitative_eval": table})

    print("Evaluation complete by LLM-Judge.Qualitative table logged to W&B.")


if __name__ == "__main__":
    main()

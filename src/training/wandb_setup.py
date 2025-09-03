import wandb
from dotenv import load_dotenv
from src.utils.config_loader import load_config
import os

load_dotenv()

wandb_cfg = load_config()["fine_tune"]["wandb"]


def init_wandb(run_name: str = None, tags: list = None, notes: str = None) -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.
    """
    wandb.login()
    wandb.init(
        project=wandb_cfg["project"],
        entity=os.getenv("WANDB_ENTITY"),
        name=run_name or wandb_cfg["run_name"],
        tags=tags,
        notes=notes,
    )
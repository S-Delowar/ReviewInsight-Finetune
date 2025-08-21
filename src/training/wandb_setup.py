import wandb
from dotenv import load_dotenv
from src.utils.config_loader import load_config
import os

load_dotenv()

wandb_cfg = load_config()["fine_tune"]["wandb"]


def init_wandb() -> None:
    """
    Initialize Weights & Biases (wandb) for experiment tracking.

    Logs into wandb using the environment token and starts a run
    with project, entity, and run name from config.
    """
    wandb.login()
    wandb.init(
        project=wandb_cfg["project"],
        entity=os.getenv("WANDB_ENTITY"),
        name=wandb_cfg["run_name"]
    )

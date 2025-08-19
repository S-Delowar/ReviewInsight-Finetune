import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yml") -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML config file.
    
    Returns:
        dict: Parsed configuration as dictionary.
    """
    project_root = Path(__file__).resolve().parents[2]
    config_file = project_root / config_path  
      
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing config file: {e}")
    
    return config

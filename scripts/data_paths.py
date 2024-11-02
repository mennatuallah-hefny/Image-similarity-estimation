import json
from pathlib import Path

def load_config(config_file='config.json'):
    """Load configuration from a JSON file."""
    with open(config_file) as file:
        return json.load(file)

def get_data_paths(data_path):
    """
    Constructs and returns paths for cache and image directories.
    
    :param data_path: Base data path.
    :return: Dictionary containing paths.
    """
    cache_dir = Path(data_path) / ".keras"
    anchor_images_path = cache_dir / "left"
    positive_images_path = cache_dir / "right"
    
    return {
        "cache_dir": cache_dir,
        "anchor_images_path": anchor_images_path,
        "positive_images_path": positive_images_path
    }

# Load configuration
config = load_config()

# Retrieve paths
paths = get_data_paths(config["DATA_PATH"])

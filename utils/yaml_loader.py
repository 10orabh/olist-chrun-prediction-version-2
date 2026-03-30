import yaml
import os

def load_yaml(path_to_yaml: str) -> dict:
    """Reads a YAML file and returns its content as a dictionary.
    Args:
        path_to_yaml (str): The file path to the YAML file.
    Returns:
        dict: The content of the YAML file as a dictionary."""
    if not os.path.exists(path_to_yaml):
        raise FileNotFoundError(f"YAML file missing at path: {path_to_yaml}")

    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            return content
    except Exception as e:
        raise ValueError(f"Error reading YAML file: {e}")
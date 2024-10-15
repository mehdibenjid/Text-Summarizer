import os
from box.exceptions import BoxValueError
import yaml
from textSummarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns a ConfigBox type
    
    Args:
    path_to_yaml: Path to the yaml file
    
    Raises:
        ValueError: If the yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file: {path_to_yaml} is empty")
    except Exception as e:
        logger.error(f"Error loading yaml file: {path_to_yaml}")
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """
    Creates directories if they do not exist
    
    Args:
    path_to_directories: List of paths to directories
    verbose: Whether to print output or not
    """
    for path in path_to_directories:
        if not os.path.exists(path):
            os.makedirs(path)
            if verbose:
                logger.info(f"Directory: {path} created successfully")
        else:
            if verbose:
                logger.info(f"Directory: {path} already exists")

@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB

    Args:
        path (Path): Path to file

    Returns:
        str: Size of file in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    return f"~{size_in_kb} KB"

"""
Utility Functions for Human Activity Recognition

This module provides utility functions for logging, saving/loading objects, and loading configuration files.

Functions:
- setup_logging(): Set up logging to a file.
- save_pickle(): Save an object to a pickle file.
- load_pickle(): Load an object from a pickle file.
- load_config(): Load configuration from a YAML file.
"""

import os
import joblib
import yaml
import logging


def setup_logging(log_file, log_level=logging.INFO, log_format=None):
    """
    Set up logging to a file.

    Args:
        log_file (str): Path to the log file.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): Logging format string. If None, a default format is used.

    Returns:
        None
    """
    if log_format is None:
        log_format = "%(asctime)s - %(levelname)s - %(message)s"

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format=log_format
    )
    logging.info("Logging setup complete.")


def save_pickle(obj, file_path):
    """
    Save an object to a pickle file.

    Args:
        obj: The object to save.
        file_path (str): Path to the output pickle file.

    Returns:
        None
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save object to {file_path}: {e}")
        raise


def load_pickle(file_path):
    """
    Load an object from a pickle file.

    Args:
        file_path (str): Path to the input pickle file.

    Returns:
        The loaded object.
    """
    try:
        obj = joblib.load(file_path)
        logging.info(f"Object loaded from {file_path}.")
        return obj
    except Exception as e:
        logging.error(f"Failed to load object from {file_path}: {e}")
        raise


def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration.
    """
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            logging.info(f"Configuration loaded from {config_file}.")
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration from {config_file}: {e}")
        raise

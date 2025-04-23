"""
Activity Recognition Inference

This module performs activity recognition based on the trained model. Given new sensor data,
the trained model is used to predict the activity being performed.

Functions:
- load_model(): Loads the pre-trained model for inference.
- preprocess_input(): Preprocesses the input data before passing it to the model.
- recognize_activity(): Recognizes the activity from the input data using the trained model.
- display_result(): Displays the predicted activity.
"""

import logging
import os

import pandas as pd

from src.project_utils import save_pickle, load_config, load_pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_model(config):
    """
    Load the pre-trained model and preprocessing objects.

    Args:
        config (dict): Configuration dictionary containing paths to the model and preprocessing objects.

    Returns:
        model: Pre-trained model.
        scaler: Scaler object used for feature scaling.
        pca: PCA object used for dimensionality reduction.
        encoder: Encoder object used for label encoding.
    """
    try:
        model = load_pickle(config["models"]["har_model"])
        scaler = load_pickle(config["models"]["scaler"])
        pca = load_pickle(config["models"]["pca"])
        encoder = load_pickle(config["models"]["encoder"])
        logging.info("Model and preprocessing objects loaded successfully.")
        return model, scaler, pca, encoder
    except Exception as e:
        logging.error(f"Failed to load model or preprocessing objects: {e}")
        raise


def preprocess_input(data, scaler, pca, training_features):
    """
    Preprocess the input data using the loaded scaler and PCA objects.

    Args:
        data (pd.DataFrame): Input data to be preprocessed.
        scaler: Scaler object for feature scaling.
        pca: PCA object for dimensionality reduction.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        data = data[training_features]
        data_scaled = scaler.transform(data)
        data_pca = pca.transform(data_scaled)
        logging.info("Input data preprocessed successfully.")
        return data_pca
    except Exception as e:
        logging.error(f"Failed to preprocess input data: {e}")
        raise


def recognize_activity(data, model, encoder):
    """
    Recognize the activity from the input data using the trained model.

    Args:
        data (pd.DataFrame): Preprocessed input data.
        model: Pre-trained model.
        encoder: Encoder object for decoding the predicted activity.

    Returns:
        np.array: Predicted activities.
    """
    try:
        prediction = model.predict(data)
        activity = encoder.inverse_transform(prediction)
        logging.info("Activity recognized successfully.")
        return activity
    except Exception as e:
        logging.error(f"Failed to recognize activity: {e}")
        raise


def display_result(activity, dataset_name):
    """
    Display the predicted activity.

    Args:
        activity (np.array): Predicted activities.
        dataset_name (str): Name of the dataset (e.g., "Test", "New Unseen Data").
    """
    print(f"\nPredicted activities for {dataset_name}:")
    print(activity)


def load_data(data_path):
    """
    Load input data for inference.

    Args:
        data_path (str): Path to the input data CSV file.

    Returns:
        pd.DataFrame: Loaded input data.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data from {data_path}: {e}")
        raise


def main():
    """
    Main function to perform activity recognition inference.
    """
    # Load configuration
    config = load_config("config/project_configuration.yml")

    # Load model and preprocessing objects
    model, scaler, pca, encoder = load_model(config)
    training_features = load_pickle(config["models"]["training_features"])

# Perform inference on test data (unseen data)
    test_path = config["data"]["test_path"]
    test_data = load_data(test_path)
    test_cleaned = test_data.drop(columns=['subject', 'Activity'], errors='ignore')
    processed_test_data = preprocess_input(test_cleaned, scaler, pca, training_features)

    # Recognize activity for test data
    test_activity = recognize_activity(processed_test_data, model, encoder)
    display_result(test_activity, "Test Data")

    new_data_path = config["data"].get("new_data_path")
    if new_data_path and os.path.exists(new_data_path):  # Check if file exists
        try:
            new_data = load_data(new_data_path)
            new_cleaned = new_data.drop(columns=['subject', 'Activity'], errors='ignore')
            processed_new_data = preprocess_input(new_cleaned, scaler, pca, training_features)

            # Recognize activity for new unseen data
            new_activity = recognize_activity(processed_new_data, model, encoder)
            display_result(new_activity, "New Unseen Data")
        except Exception as e:
            logging.error(f"Failed to process new unseen data: {e}")
    else:
        logging.warning(f"New data file not found at {new_data_path}. Skipping inference on new data.")

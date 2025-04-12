# main.py
import logging

from src.eda import perform_eda
import pandas as pd
from src.project_utils import setup_logging, load_config
from src.preprocessing import data_preprocessing
from src.strategies.CNNStrategy import CNNStrategy
from src.strategies.LSTMStrategy import LSTMStrategy
from src.strategies.ModelContext import ModelContext
from src.strategies.RandomForestStrategy import RandomForestStrategy
from src.inference import main as inference_main
from src.strategies.SVMStrategy import SVMStrategy
from src.strategies.XGBoostStrategy import XGBoostStrategy


def main():
    # Load configuration
    config = load_config("config/project_configuration.yml")

    # Set up logging
    setup_logging(config["logging"]["log_file"], log_level=config["logging"]["log_level"])

    train_path = config["data"]["train_path"]
    test_path = config["data"]["test_path"]
    # Load raw data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    logging.info("Data loaded successfully.")

    # Step 1: Perform Exploratory Data Analysis (EDA)
    logging.info("Starting Exploratory Data Analysis...")
    perform_eda(train, test, config)
    logging.info("EDA completed.")

    # Step 2: Preprocess the data
    logging.info("Starting data preprocessing...")
    X_train, y_train, X_test, y_test, scaler, pca, encoder = data_preprocessing(
        train, test, config
    )
    logging.info("Data preprocessing completed.")

    # Step 3: Train and evaluate the model
    logging.info("Starting model training and evaluation...")
    for algorithm_name in config["algorithms"]:
        logging.info(f"\n--- Running {algorithm_name} Model ---")

        # Select the strategy based on the algorithm name
        if algorithm_name == "RandomForest":
            strategy = RandomForestStrategy()
        elif algorithm_name == "SVM":
            strategy = SVMStrategy()
        elif algorithm_name == "XGBoost":
            strategy = XGBoostStrategy()
        # elif algorithm_name == "CNN":
        #     strategy = CNNStrategy()
        # elif algorithm_name == "LSTM":
        #     strategy = LSTMStrategy()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        # Create the context with the selected strategy
        context = ModelContext(strategy)

        # Train and evaluate the model
        model = context.model_training_and_evaluation(algorithm_name, X_train, y_train, X_test, y_test, config)

    logging.info("Model training and evaluation completed.")

    # model = model_training_and_evaluation(X_train, y_train, X_test, y_test, config)
    #
    # # Save the trained model
    # save_model(model, config["models"]["har_model"])
    # logging.info("Model saved.")

    logging.info("Model training and evaluation completed.")

    # Step 4: Perform inference on test data and new unseen data
    logging.info("Starting inference...")
    inference_main()
    logging.info("Inference completed.")


if __name__ == "__main__":
    main()

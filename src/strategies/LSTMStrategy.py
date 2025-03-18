# src/strategies/LSTMStrategy.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
import logging
from src.strategies.ModelStrategy import ModelStrategy

class LSTMStrategy(ModelStrategy):
    """
    Strategy for training and evaluating an LSTM model.
    """

    def train(self, X_train, y_train, X_test, y_test, config, algorithm_name="LSTM"):
        """
        Train, evaluate, and save an LSTM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "LSTM").

        Returns:
            model: Trained LSTM model.
        """
        try:
            logging.info("Training LSTM model...")

            # Build the LSTM model
            model = self.build_model(config)
            logging.info("LSTM model initialized.")

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logging.info("Model compiled.")

            # Train the model
            model.fit(
                X_train, y_train,
                epochs=config["models"]["epochs"],
                batch_size=config["models"]["batch_size"],
                validation_data=(X_test, y_test)
            )
            logging.info("Model trained.")

            # Evaluate the model
            loss, accuracy = model.evaluate(X_test, y_test)
            logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")

            # Save the model
            model_path = f"models/{algorithm_name}_model.h5"
            model.save(model_path)
            logging.info(f"Model saved to {model_path}.")

            return model

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def build_model(self, config):
        """
        Build an LSTM model based on the configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            model: Initialized LSTM model.
        """
        model = Sequential([
            Embedding(input_dim=config["models"]["vocab_size"], output_dim=config["models"]["embedding_dim"]),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),  # Add dropout
            Dense(config["models"]["num_classes"], activation='softmax')
        ])
        return model
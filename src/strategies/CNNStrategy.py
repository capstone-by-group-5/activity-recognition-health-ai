# src/strategies/CNNStrategy.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import LearningRateScheduler
import logging
from src.strategies.ModelStrategy import ModelStrategy

class CNNStrategy(ModelStrategy):
    """
    Strategy for training and evaluating a Convolutional Neural Network (CNN) model.
    """

    def train(self, X_train, y_train, X_test, y_test, config, algorithm_name="CNN"):
        """
        Train, evaluate, and save a CNN model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "CNN").

        Returns:
            model: Trained CNN model.
        """
        try:
            logging.info("Training CNN model...")

            # Build the CNN model
            model = self.build_model(config)
            logging.info("CNN model initialized.")

            # Compile the model
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            logging.info("Model compiled.")

            # Add learning rate scheduling
            lr_scheduler = LearningRateScheduler(self.lr_schedule)
            callbacks = [lr_scheduler]

            # Train the model
            model.fit(
                X_train, y_train,
                epochs=config["models"]["epochs"],
                batch_size=config["models"]["batch_size"],
                validation_data=(X_test, y_test),
                callbacks=callbacks
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
        Build a CNN model based on the configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            model: Initialized CNN model.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=config["models"]["input_shape"]),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(config["models"]["num_classes"], activation='softmax')
        ])
        return model

    def lr_schedule(self, epoch):
        """
        Learning rate schedule for the CNN model.

        Args:
            epoch (int): Current epoch.

        Returns:
            lr (float): Learning rate for the epoch.
        """
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 10
        lr = initial_lr * (drop ** (epoch // epochs_drop))
        return lr
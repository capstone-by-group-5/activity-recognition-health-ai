# src/strategies/base_strategy.py
import os
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from src.project_utils import save_pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelStrategy:
    """
    Base class for all model strategies.
    Contains common functionality like evaluation, visualization, and saving.
    """

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, config, algorithm_name):
        """
        Train, evaluate, and save the model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "RandomForest").

        Returns:
            model: Trained model.
        """
        try:
            # Train the model (to be implemented by child classes)
            model = self.train(X_train, y_train, X_test, y_test, config, algorithm_name)
            logging.info(f"{algorithm_name} model trained successfully.")

            # Make predictions on the test set
            y_pred = model.predict(X_test)
            logging.info("Predictions made on the test set.")

            # Evaluate the model
            self.evaluate_model(algorithm_name, y_test, y_pred)
            logging.info("Model evaluated.")

            # Save the model
            model_path = f"models/{algorithm_name}_model.pkl"
            self.save_model(model, model_path)
            logging.info(f"Model saved to {model_path}.")

            return model

        except Exception as e:
            logging.error(f"Error during model training and evaluation: {e}")
            raise

    def train(self, X_train, y_train, X_test, y_test, config, algorithm_name):
        """
        Train the model. This method should be overridden by child classes.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "RandomForest").

        Returns:
            model: Trained model.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def evaluate_model(self, algorithm_name, y_test, y_pred):
        """
        Evaluate the model's performance on the test set.

        Args:
            algorithm_name (str): Name of the algorithm.
            y_test (pd.Series): True labels.
            y_pred (pd.Series): Predicted labels.

        Returns:
            accuracy (float): Accuracy score.
            report (str): Classification report.
            cm (np.array): Confusion matrix.
        """
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Print evaluation metrics
        logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
        logging.info(f"Test F1-score: {f1:.4f}")
        logging.info(f"Test Precision: {precision:.4f}")
        logging.info(f"Test Recall: {recall:.4f}")
        logging.info("\nClassification Report:\n" + report)

        # Plot confusion matrix
        self.plot_confusion_matrix(cm, y_test)
        self.save_evaluation_metrics(algorithm_name, accuracy, f1, precision, recall, report, cm)
        return accuracy, report, cm

    def plot_confusion_matrix(self, cm, y_test):
        """
        Plot and save the confusion matrix.

        Args:
            cm (np.array): Confusion matrix.
            y_test (pd.Series): True labels.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig(f"results/confusion_matrix.png")
        plt.show()

    def save_evaluation_metrics(self, algorithm_name, accuracy, f1, precision, recall, report, cm):
        """
        Save evaluation metrics to files with algorithm name as prefix.

        Args:
            algorithm_name (str): Name of the algorithm (e.g., "RandomForest").
            accuracy (float): Accuracy score.
            f1 (float): F1-score.
            precision (float): Precision score.
            recall (float): Recall score.
            report (str): Classification report.
            cm (np.array): Confusion matrix.
        """
        os.makedirs("results", exist_ok=True)

        # Save accuracy, F1-score, precision, and recall to a single file
        metrics_file = f"results/{algorithm_name}_metrics.txt"
        with open(metrics_file, "w") as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")

        # Save the classification report to a separate file
        report_file = f"results/{algorithm_name}_classification_report.txt"
        with open(report_file, "w") as f:
            f.write(report)

        # Save the confusion matrix to a CSV file
        cm_file = f"results/{algorithm_name}_confusion_matrix.csv"
        np.savetxt(cm_file, cm, delimiter=",")

    def save_model(self, model, model_path):
        """
        Save the trained model to a file.

        Args:
            model: Trained model.
            model_path (str): Path to save the model.
        """
        save_pickle(model, model_path)
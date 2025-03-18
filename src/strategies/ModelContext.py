# src/strategies/ModelContext.py
import logging

class ModelContext:
    """
    Context class for the Strategy Pattern.
    Holds a reference to the current strategy and delegates the work to it.
    """

    def __init__(self, strategy):
        """
        Initialize the context with a specific strategy.

        Args:
            strategy: An instance of a strategy class (e.g., RandomForestStrategy).
        """
        self.strategy = strategy

    def model_training_and_evaluation(self, algorithm_name, X_train, y_train, X_test, y_test, config):
        """
        Train and evaluate the model using the current strategy.

        Args:
            algorithm_name (str): Name of the algorithm (e.g., "RandomForest").
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            model: Trained model.
        """
        try:
            logging.info(f"\n--- Running {algorithm_name} Model ---")

            # Delegate the work to the strategy
            model = self.strategy.train_and_evaluate(X_train, y_train, X_test, y_test, config, algorithm_name)
            logging.info(f"{algorithm_name} model trained and evaluated successfully.")

            return model

        except Exception as e:
            logging.error(f"Error during model training and evaluation: {e}")
            raise
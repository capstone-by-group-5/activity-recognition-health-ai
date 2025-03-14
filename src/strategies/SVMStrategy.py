# src/strategies/SVMStrategy.py
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
import logging
from src.strategies.ModelStrategy import ModelStrategy

class SVMStrategy(ModelStrategy):
    """
    Strategy for training and evaluating a Support Vector Machine (SVM) model.
    """

    def train(self, X_train, y_train, X_test, y_test, config, algorithm_name="SVM"):
        """
        Train, evaluate, and save an SVM model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "SVM").

        Returns:
            model (SVC): Trained SVM model.
        """
        try:
            logging.info("Training SVM model...")

            # Hyperparameter tuning
            logging.info("Starting hyperparameter tuning...")
            best_model = self.tune_hyperparameters(X_train, y_train)
            logging.info("Hyperparameter tuning completed.")

            # Update the config with the best hyperparameters
            config["models"]["C"] = best_model.C
            config["models"]["kernel"] = best_model.kernel

            # Initialize the model using the build_model method
            model = self.build_model(config)
            logging.info("SVM model initialized with tuned hyperparameters.")

            # Perform cross-validation
            logging.info("Performing cross-validation...")
            cv_scores = self.cross_validate(model, X_train, y_train)
            logging.info(f"Cross-validation accuracy scores: {cv_scores}")
            logging.info(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")

            # Train the model
            model.fit(X_train, y_train)
            logging.info("Model trained.")

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
            logging.error(f"Error during model training: {e}")
            raise

    def tune_hyperparameters(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            best_model: The best model with optimized hyperparameters.
        """
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        grid_search = GridSearchCV(
            SVC(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        logging.info(f"Best hyperparameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def cross_validate(self, model, X_train, y_train):
        """
        Perform cross-validation on the training data.

        Args:
            model: The model to evaluate.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.

        Returns:
            cv_scores (np.array): Cross-validation accuracy scores.
        """
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return cv_scores

    def build_model(self, config):
        """
        Build an SVM model based on the configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            model (SVC): Initialized SVM model.
        """
        model = SVC(
            C=config["models"]["C"],
            kernel=config["models"]["kernel"],
            random_state=config["models"]["random_state"],
            probability=True  # Enable probability estimates
        )
        return model
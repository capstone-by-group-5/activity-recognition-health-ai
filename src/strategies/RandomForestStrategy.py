# src/strategies/RandomForestStrategy.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import logging

from sklearn.utils import compute_class_weight

from src.strategies.ModelStrategy import ModelStrategy

class RandomForestStrategy(ModelStrategy):
    """
    Strategy for training and evaluating a Random Forest model.
    """

    def train(self, X_train, y_train, X_test, y_test, config, algorithm_name="RandomForest"):
        """
        Train, evaluate, and save a Random Forest model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "RandomForest").

        Returns:
            model (RandomForestClassifier): Trained Random Forest model.
        """
        try:
            logging.info("Training ..", )

            # Compute class weights
            class_weight_dict = self.compute_class_weights(y_train)
            config["models"]["class_weight"] = class_weight_dict
            logging.info(f"Computed class weights: {class_weight_dict}")

            # Hyperparameter tuning
            logging.info("Starting hyperparameter tuning...")
            best_model = self.tune_hyperparameters(X_train, y_train)
            logging.info("Hyperparameter tuning completed.")

            # Update the config with the best hyperparameters
            config["models"]["n_estimators"] = best_model.n_estimators
            config["models"]["max_depth"] = best_model.max_depth
            config["models"]["min_samples_split"] = best_model.min_samples_split

            # Initialize the model using the build_model method
            model = self.build_model(config)
            logging.info("Random Forest model initialized with tuned hyperparameters.")

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

            # Log feature importance
            self.log_feature_importance(model, X_train.columns)
            logging.info("Feature importance logged.")

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
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
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

    def compute_class_weights(self, y):
        """
        Compute class weights based on the target labels `y`.

        Args:
            y (pd.Series): Target labels.

        Returns:
            class_weight_dict (dict): Dictionary of class weights.
        """
        classes = np.unique(y)
        class_weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        return class_weight_dict

    def build_model(self, config):
        """
        Build a Random Forest model based on the configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            model (RandomForestClassifier): Initialized Random Forest model.
        """
        model = RandomForestClassifier(
            n_estimators=config["models"]["n_estimators"],
            max_depth=config["models"]["max_depth"],
            min_samples_split=config["models"]["min_samples_split"],
            random_state=config["models"]["random_state"],
            class_weight=config["models"]["class_weight"],
            warm_start=True  # Enable incremental training
        )
        return model

    def log_feature_importance(self, model, feature_names):
        """
        Log feature importance for the Random Forest model.

        Args:
            model (RandomForestClassifier): Trained Random Forest model.
            feature_names (list): List of feature names.
        """
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        logging.info("Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{feature}: {importance:.4f}")
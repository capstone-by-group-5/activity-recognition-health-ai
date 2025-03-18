from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import logging
from src.strategies.ModelStrategy import ModelStrategy


def tune_hyperparameters(X_train, y_train):
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
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(
        XGBClassifier(random_state=42, objective="multi:softmax", eval_metric="mlogloss"),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logging.info(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


class XGBoostStrategy(ModelStrategy):
    """
    Strategy for training and evaluating an XGBoost model.
    """

    def train(self, X_train, y_train, X_test, y_test, config, algorithm_name="XGBoost"):
        """
        Train, evaluate, and save an XGBoost model.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            config (dict): Configuration dictionary containing model parameters.
            algorithm_name (str): Name of the algorithm (e.g., "XGBoost").

        Returns:
            model (XGBClassifier): Trained XGBoost model.
        """
        try:
            logging.info("Training XGBoost model...")

            # Hyperparameter tuning
            logging.info("Starting hyperparameter tuning...")
            best_model = tune_hyperparameters(X_train, y_train)
            logging.info("Hyperparameter tuning completed.")

            # Update the config with the best hyperparameters
            config["models"]["n_estimators"] = best_model.n_estimators
            config["models"]["max_depth"] = best_model.max_depth
            config["models"]["learning_rate"] = best_model.learning_rate

            # Initialize the model using the build_model method
            model = self.build_model(config)
            logging.info("XGBoost model initialized with tuned hyperparameters.")

            # Perform cross-validation (disable early stopping)
            logging.info("Performing cross-validation...")
            cv_scores = self.cross_validate(model, X_train, y_train)
            logging.info(f"Cross-validation accuracy scores: {cv_scores}")
            logging.info(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")

            # Train the model with early stopping
            logging.info("Training model with early stopping...")
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],  # Evaluation set for early stopping
                verbose=True                   # Print progress
            )
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
        # Disable early stopping for cross-validation
        model.set_params(early_stopping_rounds=None)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return cv_scores

    def build_model(self, config):
        """
        Build an XGBoost model based on the configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.

        Returns:
            model (XGBClassifier): Initialized XGBoost model.
        """
        model = XGBClassifier(
            n_estimators=config["models"]["n_estimators"],
            max_depth=config["models"]["max_depth"],
            learning_rate=config["models"]["learning_rate"],
            random_state=config["models"]["random_state"],
            early_stopping_rounds=10,  # Early stopping rounds
            objective="multi:softmax",  # Multi-class classification
            eval_metric="mlogloss"      # Multi-class log loss
        )
        return model

    def log_feature_importance(self, model, feature_names):
        """
        Log feature importance for the XGBoost model.

        Args:
            model (XGBClassifier): Trained XGBoost model.
            feature_names (list): List of feature names.
        """
        importance = model.feature_importances_
        feature_importance = dict(zip(feature_names, importance))
        logging.info("Feature Importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"{feature}: {importance:.4f}")
# """
# Model Training for Human Activity Recognition
#
# This module handles the training of machine learning models for activity recognition
# using the preprocessed data. It includes functions for defining, training, and saving
# the model.
#
# Functions:
# - build_model(): Defines the machine learning model architecture.
# - train_model(): Trains the model using the training data.
# - save_model(): Saves the trained model for future inference.
# - evaluate_model(): Evaluates the model's performance on the test set.
# """
#
# import os
# import logging
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, \
#     recall_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
#
# from src.project_utils import save_pickle
#
# #from project_utils import save_pickle, load_config
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#
# def model_training_and_evaluation(algorithm_name, X_train, y_train, X_test, y_test, config):
#     """
#     Train and evaluate a Random Forest model.
#
#     Args:
#         X_train (pd.DataFrame): Training features.
#         y_train (pd.Series): Training labels.
#         X_test (pd.DataFrame): Testing features.
#         y_test (pd.Series): Testing labels.
#         config (dict): Configuration dictionary containing model parameters.
#
#     Returns:
#         model (RandomForestClassifier): Trained Random Forest model.
#     """
#     try:
#         logging.info(f"\n--- Training and Evaluating {algorithm_name} Model ---")
#
#         # Hyperparameter tuning
#         logging.info("Starting hyperparameter tuning...")
#         best_model = tune_hyperparameters(X_train, y_train)
#         logging.info("Hyperparameter tuning completed.")
#
#         # Initialize the model with the best hyperparameters
#         model = RandomForestClassifier(
#             n_estimators=best_model.n_estimators,
#             max_depth=best_model.max_depth,
#             min_samples_split=best_model.min_samples_split,
#             random_state=config["models"]["random_state"],
#             class_weight=config["models"]["class_weight"],
#         )
#         logging.info(f"{algorithm_name} model initialized with tuned hyperparameters.")
#
#         # Perform cross-validation
#         logging.info("Performing cross-validation...")
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config["models"]["random_state"])
#         cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
#         logging.info(f"Cross-validation accuracy scores: {cv_scores}")
#         logging.info(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")
#
#         # Initialize the model
#         #model = build_model(config)
#         logging.info(f"{algorithm_name} model trained on the full training set.")
#
#         # Train the model
#         model.fit(X_train, y_train)
#         logging.info("Model trained.")
#
#         # Make predictions
#         y_pred = model.predict(X_test)
#         logging.info("Predictions made on the test set.")
#
#         # Evaluate model performance
#         evaluate_model(algorithm_name, y_test, y_pred)
#         logging.info("Model evaluated.")
#
#         # Save evaluation metrics
#        # save_evaluation_metrics(accuracy, report, cm)
#
#         logging.info("Evaluation metrics saved.")
#
#         return model
#
#     except Exception as e:
#         logging.error(f"Error during model training and evaluation: {e}")
#         raise
#
#
# def tune_hyperparameters(X_train, y_train):
#     """
#     Perform hyperparameter tuning using GridSearchCV.
#
#     Args:
#         X_train (pd.DataFrame): Training features.
#         y_train (pd.Series): Training labels.
#
#     Returns:
#         best_model: The best model with optimized hyperparameters.
#     """
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [10, 20, 30],
#         'min_samples_split': [2, 5, 10]
#     }
#     grid_search = GridSearchCV(
#         RandomForestClassifier(random_state=42),
#         param_grid,
#         cv=5,
#         scoring='accuracy',
#         n_jobs=-1
#     )
#     grid_search.fit(X_train, y_train)
#     logging.info(f"Best hyperparameters: {grid_search.best_params_}")
#     return grid_search.best_estimator_
#
#
# def build_model(config):
#     """
#     Build a Random Forest model based on the configuration.
#
#     Args:
#         config (dict): Configuration dictionary containing model parameters.
#
#     Returns:
#         model (RandomForestClassifier): Initialized Random Forest model.
#     """
#     model = RandomForestClassifier(
#         n_estimators=config["models"]["n_estimators"],
#         max_depth=config["models"]["max_depth"],
#         min_samples_split=config["models"]["min_samples_split"],
#         random_state=config["models"]["random_state"],
#         class_weight=config["models"]["class_weight"],
#     )
#     return model
#
#
# def evaluate_model(algorithm_name, y_test, y_pred):
#     """
#     Evaluate the model's performance on the test set.
#
#     Args:
#         y_test (pd.Series): True labels.
#         y_pred (pd.Series): Predicted labels.
#
#     Returns:
#         accuracy (float): Accuracy score.
#         report (str): Classification report.
#         cm (np.array): Confusion matrix.
#     """
#     accuracy = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     precision = precision_score(y_test, y_pred, average='weighted')
#     recall = recall_score(y_test, y_pred, average='weighted')
#     report = classification_report(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#
#     # Print evaluation metrics
#     logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
#     logging.info(f"Test F1-score: {f1:.4f}")
#     logging.info(f"Test Precision: {precision:.4f}")
#     logging.info(f"Test Recall: {recall:.4f}")
#     logging.info("\nClassification Report:\n" + report)
#
#     # Plot confusion matrix
#     plot_confusion_matrix(cm, y_test)
#     save_evaluation_metrics(algorithm_name, accuracy, f1, precision, recall, report, cm)
#     return accuracy, report, cm
#
#
# def plot_confusion_matrix(cm, y_test):
#     """
#     Plot and save the confusion matrix.
#
#     Args:
#         cm (np.array): Confusion matrix.
#         y_test (pd.Series): True labels.
#     """
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.savefig("results/confusion_matrix.png")
#     plt.show()
#
#
# def save_evaluation_metrics(algorithm_name, accuracy, f1, precision, recall, report, cm):
#     """
#     Save evaluation metrics to files with algorithm name as prefix.
#
#     Args:
#         algorithm_name (str): Name of the algorithm (e.g., "RandomForest").
#         accuracy (float): Accuracy score.
#         f1 (float): F1-score.
#         precision (float): Precision score.
#         recall (float): Recall score.
#         report (str): Classification report.
#         cm (np.array): Confusion matrix.
#     """
#     # os.makedirs("results", exist_ok=True)
#     # with open("results/accuracy.txt", "w") as f:
#     #     f.write(f"Accuracy: {accuracy * 100:.2f}%")
#     # with open("results/classification_report.txt", "w") as f:
#     #     f.write(report)
#     # np.savetxt("results/confusion_matrix.csv", cm, delimiter=",")
#     #
#     # os.makedirs("results", exist_ok=True)
#
#     # Save accuracy, F1-score, precision, and recall to a single file
#     metrics_file = f"results/{algorithm_name}_metrics.txt"
#     with open(metrics_file, "w") as f:
#         f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
#         f.write(f"F1-score: {f1:.4f}\n")
#         f.write(f"Precision: {precision:.4f}\n")
#         f.write(f"Recall: {recall:.4f}\n")
#
#     # Save the classification report to a separate file
#     report_file = f"results/{algorithm_name}_classification_report.txt"
#     with open(report_file, "w") as f:
#         f.write(report)
#
#     # Save the confusion matrix to a CSV file
#     cm_file = f"results/{algorithm_name}_confusion_matrix.csv"
#     np.savetxt(cm_file, cm, delimiter=",")
#
#
# def save_model(model, model_path):
#     """
#     Save the trained model to a file.
#
#     Args:
#         model: Trained model.
#         model_path (str): Path to save the model.
#     """
#     save_pickle(model, model_path)

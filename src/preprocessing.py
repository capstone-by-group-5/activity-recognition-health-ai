"""
Data Preprocessing for Human Activity Recognition

This module handles the preprocessing of raw sensor data collected from wearable devices.
The main functions include cleaning, normalization, feature extraction, and data splitting
for model training and evaluation.

Functions:
- clean_data(): Cleans the raw sensor data by handling missing values, removing outliers, etc.
- normalize_data(): Normalizes the data to a common scale.
- extract_features(): Extracts relevant features from the raw sensor data.
- split_data(): Splits the dataset into training, validation, and test sets.
"""
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
#from project_utils import save_pickle  # Ensure this import works
import matplotlib.pyplot as plt
from src.project_utils import save_pickle, load_config

import sys
import os

print("Python Path:", sys.path)
print("Current Working Directory:", os.getcwd())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def data_preprocessing(train, test, config):
    """
    Preprocess data: scaling, PCA, and feature selection.

    Args:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the test dataset CSV file.
        config (dict): Configuration dictionary containing preprocessing settings.

    Returns:
        X_resampled (pd.DataFrame): Resampled training features.
        y_resampled (np.array): Resampled training labels.
        X_test_scaled (pd.DataFrame): Preprocessed test features.
        y_test (np.array): Test labels.
        scaler: Fitted scaler object.
        pca: Fitted PCA object.
        encoder: Fitted label encoder.
    """
    try:
        logging.info("\n--- Preprocessing Data ---")

        # Handle missing values and duplicates
        train = handle_missing_and_duplicates(train)
        test = handle_missing_and_duplicates(test)

        # Remove highly correlated features
        correlation_threshold = config["preprocessing"]["correlation_threshold"]

        X_train, X_test = remove_highly_correlated_features(train, test, correlation_threshold)
        logging.info("Highly correlated features removed.")

        # Remove high VIF features
        vif_threshold = config["preprocessing"]["vif_threshold"]
        X_train, X_test, vif_data = remove_high_vif_features(X_train, X_test, vif_threshold)
        logging.info("High VIF features removed.")

        # Drop non-numeric columns for preprocessing
        X_train_cleaned = X_train.drop(columns=['subject', 'Activity'], errors='ignore')
        X_test_cleaned = X_test.drop(columns=['subject', 'Activity'], errors='ignore')

        # Save the list of features used for training
        training_features = X_train_cleaned.columns.tolist()
        save_pickle(training_features, config["models"]["training_features"])
        logging.info("Training features saved.")

        # Encode target variable
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(X_train['Activity'])
        y_test = encoder.transform(X_test['Activity'])
        logging.info("Target variable encoded.")

        # Apply scaling and PCA
        X_train_scaled, X_test_scaled, scaler, pca = preprocess_data(X_train_cleaned, X_test_cleaned)
        logging.info("Data scaled and PCA applied.")

        # Check class imbalance before SMOTE
        check_class_imbalance(y_train, encoder)

        # Handle class imbalance
        X_resampled, y_resampled = handle_class_imbalance(X_train_scaled, y_train)
        logging.info("Class imbalance handled using SMOTE.")

        # Save processed data
        save_pickle(X_resampled, config["processed_data"]["train"])
        save_pickle(X_test_scaled, config["processed_data"]["test"])
        save_pickle(scaler, config["models"]["scaler"])
        save_pickle(pca, config["models"]["pca"])
        save_pickle(encoder, config["models"]["encoder"])
        logging.info("Processed data and objects saved.")

        return X_resampled, y_resampled, X_test_scaled, y_test, scaler, pca, encoder

    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def handle_missing_and_duplicates(train):
    """
    Handle missing values and remove duplicate rows.
    """
    logging.info("\n--- Handling Missing Data and Duplicates ---")

    # Fill missing values with mean for numerical columns
    for column in train.select_dtypes(include=[np.number]).columns:
        train[column] = train[column].fillna(train[column].mean())

    # Fill missing values with mode for categorical columns
    for column in train.select_dtypes(include=['object']).columns:
        train[column] = train[column].fillna(train[column].mode()[0])

    # Remove duplicates
    initial_shape = train.shape[0]
    train = train.drop_duplicates()
    logging.info(f"Removed {initial_shape - train.shape[0]} duplicate rows.")

    return train


def remove_highly_correlated_features(train, test, threshold=0.97):
    """
    Remove features with correlation above the threshold.

    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        threshold (float): Correlation threshold for feature removal.

    Returns:
        X_train (pd.DataFrame): Training data with highly correlated features removed.
        X_test (pd.DataFrame): Test data with highly correlated features removed.
    """
    numeric_train = train.select_dtypes(include=['float64', 'int64'])
    X_train = numeric_train.drop(columns=['subject', 'Activity'], errors='ignore')

    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    logging.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")

    X_train = train.drop(columns=to_drop)
    X_test = test.drop(columns=to_drop, errors='ignore')

    return X_train, X_test


def remove_high_vif_features(train, test, threshold=10):
    """
    Remove features with high VIF.

    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.
        threshold (float): VIF threshold for feature removal.

    Returns:
        train_reduced (pd.DataFrame): Training data with high VIF features removed.
        test_reduced (pd.DataFrame): Test data with high VIF features removed.
        vif_data (pd.DataFrame): DataFrame containing VIF values for each feature.
    """
    numeric_train = train.select_dtypes(include=['float64', 'int64'])
    X = numeric_train.drop(columns=['subject', 'Activity'], errors='ignore').copy()

    vif_values = svd_vif(X)
    vif_data = pd.DataFrame({"Feature": X.columns, "VIF": vif_values})
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    high_vif_features = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
    logging.info(f"Dropping {len(high_vif_features)} features with VIF > {threshold}: {high_vif_features}")

    train_reduced = train.drop(columns=high_vif_features, errors='ignore')
    test_reduced = test.drop(columns=high_vif_features, errors='ignore')

    return train_reduced, test_reduced, vif_data


def svd_vif(X):
    """
    Compute VIF using Singular Value Decomposition.

    Args:
        X (pd.DataFrame): Input data.

    Returns:
        vif_values (np.array): VIF values for each feature.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    vif_values = 1 / (s ** 2)
    return vif_values


def preprocess_data(X_train, X_test):
    """
    Preprocess data: scaling, PCA, and feature selection.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Test features.

    Returns:
        X_train_scaled (pd.DataFrame): Scaled and PCA-transformed training features.
        X_test_scaled (pd.DataFrame): Scaled and PCA-transformed test features.
        scaler: Fitted scaler object.
        pca: Fitted PCA object.
    """
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Preserve feature names for PCA components
    pca_columns = [f"PC{i + 1}" for i in range(X_train_pca.shape[1])]
    X_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns)
    X_test_pca = pd.DataFrame(X_test_pca, columns=pca_columns)

    return X_train_pca, X_test_pca, scaler, pca


def check_class_imbalance(y_train, encoder):
    """
    Checks and visualizes class imbalance with actual activity names.

    Args:
        y_train (np.array): Training labels.
        encoder: Fitted label encoder.
    """
    label_counts = pd.Series(y_train).value_counts()
    activity_labels = encoder.inverse_transform(label_counts.index)

    # Plot class distribution with activity names
    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=activity_labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
    plt.title("Distribution of Activity Classes")
    plt.savefig("results/class_distribution.png")
    plt.show()


def handle_class_imbalance(X_train, y_train):
    """
    Apply SMOTE to handle class imbalance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (np.array): Training labels.

    Returns:
        X_resampled (pd.DataFrame): Resampled training features.
        y_resampled (np.array): Resampled training labels.
    """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

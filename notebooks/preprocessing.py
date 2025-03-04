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
import pandas as pd
import numpy as np
import os
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define paths
data_dir = "data/raw"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def clean_data(df):
    """Cleans the raw sensor data by handling missing values, removing outliers, etc."""
    # Handle missing values (e.g., using forward fill or replacing with mean)
    df.fillna(df.mean(), inplace=True)

    # Optionally remove outliers, here we define outliers as values beyond 3 standard deviations
    for col in df.select_dtypes(include=[np.number]).columns:
        df = df[np.abs(df[col] - df[col].mean()) <= (3 * df[col].std())]

    return df

def normalize_data(df):
    """Normalizes the data to a common scale using StandardScaler."""
    scaler = StandardScaler()
    df_normalized = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df_normalized[col] = scaler.fit_transform(df[[col]])
    return df_normalized

def extract_features(df):
    """Extracts statistical and FFT features from numerical columns."""
    feature_df = pd.DataFrame()
    for col in df.select_dtypes(include=[np.number]).columns:
        feature_df[f'{col}_mean'] = [df[col].mean()]
        feature_df[f'{col}_std'] = [df[col].std()]
        feature_df[f'{col}_min'] = [df[col].min()]
        feature_df[f'{col}_max'] = [df[col].max()]
        feature_df[f'{col}_median'] = [df[col].median()]
        feature_df[f'{col}_skew'] = [skew(df[col].dropna())]
        feature_df[f'{col}_kurtosis'] = [kurtosis(df[col].dropna())]
        fft_values = np.abs(fft(df[col].dropna()))
        feature_df[f'{col}_fft_energy'] = [np.sum(fft_values**2)]
    return feature_df

def split_data(df):
    """Splits the data into training, validation, and test sets."""
    X = df.drop(columns='activity_label')  # Assuming 'activity_label' column for labels
    y = df['activity_label']  # Assuming the target column is 'activity_label'

    # Split into train+validation (80%) and test (20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split train+validation into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Clean and normalize data
train_df_clean = clean_data(train_df)
test_df_clean = clean_data(test_df)

train_df_normalized = normalize_data(train_df_clean)
test_df_normalized = normalize_data(test_df_clean)

# Extract features
train_features = extract_features(train_df_normalized)
test_features = extract_features(test_df_normalized)

# Split data into training, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(train_df_normalized)

# Save to Excel
train_output = os.path.join(output_dir, "train_features.xlsx")
test_output = os.path.join(output_dir, "test_features.xlsx")
train_features.to_excel(train_output, index=False)
test_features.to_excel(test_output, index=False)

print(f"Data preprocessing complete! Files saved to: {output_dir}")
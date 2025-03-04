

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import os

# Define paths
data_dir = "data/raw"
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def extract_features(df):
    """Extracts statistical and FFT features from numerical columns."""
    feature_df = pd.DataFrame()
    for col in df.select_dtypes(include=[np.number]).columns:
        # Statistical features
        feature_df[f'{col}_mean'] = [df[col].mean()]
        feature_df[f'{col}_std'] = [df[col].std()]
        feature_df[f'{col}_min'] = [df[col].min()]
        feature_df[f'{col}_max'] = [df[col].max()]
        feature_df[f'{col}_median'] = [df[col].median()]
        feature_df[f'{col}_skew'] = [skew(df[col].dropna())]
        feature_df[f'{col}_kurtosis'] = [kurtosis(df[col].dropna())]

        # FFT-based feature
        fft_values = np.abs(fft(df[col].dropna()))  # Compute FFT and take the absolute values
        feature_df[f'{col}_fft_energy'] = [np.sum(fft_values**2)]  # Energy of the signal

    return feature_df

# Extract features for both training and test datasets
train_features = extract_features(train_df)
test_features = extract_features(test_df)

# Save to Excel
train_output = os.path.join(output_dir, "train_features.xlsx")
test_output = os.path.join(output_dir, "test_features.xlsx")
train_features.to_excel(train_output, index=False)
test_features.to_excel(test_output, index=False)

print(f"Feature extraction complete! Files saved to: {output_dir}")
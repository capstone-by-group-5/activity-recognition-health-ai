"""
Activity Recognition Inference

This module performs activity recognition based on the trained model. Given new sensor data,
the trained model is used to predict the activity being performed.

Functions:
- load_model(): Loads the pre-trained model for inference.
- preprocess_input(): Preprocesses the input data before passing it to the model.
- recognize_activity(): Recognizes the activity from the input data using the trained model.
- display_result(): Displays the predicted activity.
"""

import pandas as pd
from ydata_profiling import ProfileReport

def load_csv(file_path):
    """Loads a CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print("CSV file loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: CSV parsing issue.")
        return None

def generate_profile_report(df, output_file):
    """Generates a profile report for the DataFrame."""
    if df is not None:
        profile = ProfileReport(df, explorative=True)
        profile.to_file(output_file)
        print(f"Profiling report saved as: {output_file}")
    else:
        print("Skipping profile generation due to DataFrame loading failure.")

def main():

    csv_filename = "data/raw/train.csv"
    output_report = "docs/pandas_profiling_report.html"
    df = load_csv(csv_filename)
    
    if df is not None:
        print(df.head())
        print(df.info())
        print(df.describe())
    
    generate_profile_report(df, output_report)

if __name__ == "__main__":
    main()

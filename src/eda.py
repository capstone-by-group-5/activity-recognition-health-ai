# eda.py
import logging
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def perform_eda(train, test, config):
    """
    Perform exploratory data analysis (EDA) on the training and test datasets.

    Args:
        train : Path to the training dataset CSV file.
        test : Path to the test dataset CSV file.
        config (dict): Configuration dictionary containing EDA settings.
    """
    try:
        # Perform EDA
        EDA_check(train, test, config)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


def EDA_check(train, test, config):
    """Perform exploratory data analysis."""
    logging.info("\n--- Performing EDA ---")
    data_overview(train, test)
    basic_statistics(train)
    class_distribution(train, config)
    missing_data(train)
    feature_relationships(train, config)
    feature_distributions(train, config)


# Sub Methods for EDA
def data_overview(train, test):
    """Display basic information about the datasets."""
    logging.info("\n--- TRAIN DATA OVERVIEW ---")
    logging.info(train.info())
    logging.info(train.head())
    logging.info(f"Shape: {train.shape}")
    logging.info("\n--- TEST DATA OVERVIEW ---")
    logging.info(test.info())
    logging.info(test.head())
    logging.info(f"Shape: {test.shape}")


def basic_statistics(train):
    """Display basic statistics and column information."""
    logging.info("\n--- BASIC STATISTICS ---")
    logging.info(train.describe())
    logging.info("\nUnique values per column:")
    logging.info(train.nunique())

    cat_cols = train.select_dtypes(include=['object']).columns
    num_cols = train.select_dtypes(include=['float64', 'int64']).columns
    logging.info(f"\nCategorical Columns: {cat_cols}")
    logging.info(f"\nNumerical Columns: {num_cols}")


def class_distribution(train, config):
    """Plot the distribution of target classes."""
    plt.figure(figsize=config["eda"]["class_distribution_figsize"])
    sns.countplot(x='Activity', data=train)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.savefig("results/class_distribution.png")
    plt.show()


def missing_data(train):
    """Check for missing values and duplicates."""
    logging.info("\n--- MISSING DATA ---")
    missing = train.isnull().sum()
    logging.info(missing[missing > 0])

    logging.info("\n--- DUPLICATES ---")
    duplicates = train.duplicated().sum()
    logging.info(f"Number of duplicate rows: {duplicates}")


def feature_relationships(train, config):
    """Plot feature correlation matrix."""
    numeric_train = train.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_train.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, linewidths=0.1)
    plt.title('Feature Correlation Matrix')
    plt.savefig("results/feature_correlation_matrix.png")
    plt.show()


def feature_distributions(train, config):
    """
    Plot feature distributions in larger batches to reduce the number of graphs.

    Args:
        train (pd.DataFrame): The dataset to visualize.
        config (dict): Configuration dictionary for plot settings.
    """
    num_features = train.shape[1]
    features_per_plot = config["eda"]["features_per_plot"]
    num_plots = int(np.ceil(num_features / features_per_plot))

    for i in range(num_plots):
        start_col = i * features_per_plot
        end_col = min((i + 1) * features_per_plot, num_features)
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))
        axes = axes.ravel()

        for j, col in enumerate(train.columns[start_col:end_col]):
            sns.histplot(train[col], ax=axes[j], kde=True)
            axes[j].set_title(col)
            axes[j].set_xticks([])

        plt.tight_layout()
        plt.savefig(f"results/feature_distribution_{i + 1}.png")
        plt.show()

# def main():
#     # Load configuration
#     config = load_config("config/project_configuration.yml")
#
#     # Ensure the results directory exists
#     os.makedirs("results", exist_ok=True)
#
#     # Perform EDA
#     perform_eda(config["data"]["train_path"], config["data"]["test_path"], config)
#
#
# if __name__ == "__main__":
#     main()

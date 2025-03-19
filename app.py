import os
import logging
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import streamlit as st

# Dataset paths

train_path = "data/raw/train.csv"
test_path = "data/raw/test.csv"

# Define target_names
target_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

# Load Data
def load_data(train_path, test_path):
    """Load train and test datasets."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# EDA Check
def EDA_check(train, test):
    """Perform exploratory data analysis."""
    st.write("\n--- Performing EDA ---")
    data_overview(train, test)
    basic_statistics(train)
    class_distribution(train)
    missing_data(train)
    feature_relationships(train)
   # feature_distributions(train)

# Sub Methods for EDA
def data_overview(train, test):
    """Display basic information about the datasets."""
    st.write("\n--- TRAIN DATA OVERVIEW ---")
    st.write(train.info())
    st.write(train.head())
    st.write("Shape:", train.shape)

    st.write("\n--- TEST DATA OVERVIEW ---")
    st.write(test.info())
    st.write(test.head())
    st.write("Shape:", test.shape)

def basic_statistics(train):
    """Display basic statistics and column information."""
    st.write("\n--- BASIC STATISTICS ---")
    st.write(train.describe())
    st.write("\nUnique values per column:")
    st.write(train.nunique())

    cat_cols = train.select_dtypes(include=['object']).columns
    num_cols = train.select_dtypes(include=['float64', 'int64']).columns
    st.write("\nCategorical Columns:", cat_cols)
    st.write("\nNumerical Columns:", num_cols)

def class_distribution(train):
    """Plot the distribution of target classes."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Activity', data=train)
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    st.pyplot(plt)

def missing_data(train):
    """Check for missing values and duplicates."""
    st.write("\n--- MISSING DATA ---")
    missing = train.isnull().sum()
    st.write(missing[missing > 0])

    st.write("\n--- DUPLICATES ---")
    duplicates = train.duplicated().sum()
    st.write(f"Number of duplicate rows: {duplicates}")

def feature_relationships(train):
    """Plot feature correlation matrix."""
    numeric_train = train.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_train.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, linewidths=0.1)
    plt.title('Feature Correlation Matrix')
    st.pyplot(plt)

def feature_distributions(train):
    """Plot feature distributions."""
    num_features = train.shape[1]
    features_per_plot = 100  # Number of features per subplot
    num_plots = int(np.ceil(num_features / features_per_plot))

    # Plot histograms in batches
    for i in range(num_plots):
        start_col = i * features_per_plot
        end_col = min((i + 1) * features_per_plot, num_features)

        train.iloc[:, start_col:end_col].hist(figsize=(16, 12), bins=30)
        plt.suptitle(f'Feature Distribution (Columns {start_col+1} to {end_col})')
        st.pyplot(plt)

    # Plot boxplots in smaller groups
    boxplot_features = 100  # Number of features per boxplot
    num_boxplots = int(np.ceil(num_features / boxplot_features))

    for i in range(num_boxplots):
        start_col = i * boxplot_features
        end_col = min((i + 1) * boxplot_features, num_features)

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=train.iloc[:, start_col:end_col])
        plt.title(f'Boxplot of Features {start_col+1} to {end_col}')
        plt.xticks(rotation=90)
        st.pyplot(plt)

# Data Preprocessing
def data_preprocessing(train, test):
    """Preprocess data: scaling, PCA, and feature selection."""
    st.write("\n--- Preprocessing Data ---")
    X_train, X_test = remove_highly_correlated_features(train, test)
    X_train, X_test, vif_data = remove_high_vif_features(X_train, X_test)

    # Separate features and target
    X_train_cleaned = X_train.drop(columns=['Activity'], errors='ignore')
    X_test_cleaned = X_test.drop(columns=['Activity'], errors='ignore')

    # Encode the 'subject' column using OneHotEncoding
    subject_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_subject_encoded = subject_encoder.fit_transform(X_train[['subject']])
    X_test_subject_encoded = subject_encoder.transform(X_test[['subject']])

    # Drop the original 'subject' column
    X_train_cleaned = X_train_cleaned.drop(columns=['subject'], errors='ignore')
    X_test_cleaned = X_test_cleaned.drop(columns=['subject'], errors='ignore')

    # Apply scaling and PCA
    X_train_scaled, X_test_scaled, scaler, pca = preprocess_data(X_train_cleaned, X_test_cleaned)

    # Combine scaled features with encoded subject data
    X_train_final = np.hstack((X_train_scaled, X_train_subject_encoded))
    X_test_final = np.hstack((X_test_scaled, X_test_subject_encoded))

    # Encode target variable
    activity_encoder = LabelEncoder()
    y_train = activity_encoder.fit_transform(X_train['Activity'])
    y_test = activity_encoder.transform(X_test['Activity'])

    # Check class imbalance before SMOTE
    check_class_imbalance(y_train, activity_encoder)

    # Handle class imbalance
    X_resampled, y_resampled = handle_class_imbalance(X_train_final, y_train)

    return X_resampled, y_resampled, X_test_final, y_test, scaler, pca, activity_encoder, subject_encoder

# Remove correlated Features
def remove_highly_correlated_features(train, test, threshold=0.97):
    """Remove features with correlation above the threshold."""
    activity_train = train['Activity'] if 'Activity' in train.columns else None
    subject_train = train['subject'] if 'subject' in train.columns else None
    activity_test = test['Activity'] if 'Activity' in test.columns else None
    subject_test = test['subject'] if 'subject' in test.columns else None

    numeric_train = train.select_dtypes(include=['float64', 'int64'])
    X_train = numeric_train.drop(columns=['subject', 'Activity'], errors='ignore')

    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    st.write(f"Removing {len(to_drop)} highly correlated features: {to_drop}")

    X_train = X_train.drop(columns=to_drop)
    X_test = test.drop(columns=to_drop, errors='ignore')

    if activity_train is not None:
        X_train['Activity'] = activity_train
    if subject_train is not None:
        X_train['subject'] = subject_train
    if activity_test is not None:
        X_test['Activity'] = activity_test
    if subject_test is not None:
        X_test['subject'] = subject_test

    return X_train, X_test

# Remove High VIF Features
def remove_high_vif_features(train, test, threshold=10):
    """Remove features with high VIF."""
    activity_train = train['Activity'] if 'Activity' in train.columns else None
    subject_train = train['subject'] if 'subject' in train.columns else None
    activity_test = test['Activity'] if 'Activity' in test.columns else None
    subject_test = test['subject'] if 'subject' in test.columns else None

    numeric_train = train.select_dtypes(include=['float64', 'int64'])
    X = numeric_train.drop(columns=['subject', 'Activity'], errors='ignore').copy()

    vif_values = svd_vif(X)
    vif_data = pd.DataFrame({"Feature": X.columns, "VIF": vif_values})
    vif_data = vif_data.sort_values(by="VIF", ascending=False)

    high_vif_features = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
    st.write(f"Dropping {len(high_vif_features)} features with VIF > {threshold}: {high_vif_features}")

    train_reduced = train.drop(columns=high_vif_features, errors='ignore')
    test_reduced = test.drop(columns=high_vif_features, errors='ignore')

    return train_reduced, test_reduced, vif_data

def svd_vif(X):
    """Compute VIF using Singular Value Decomposition."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    vif_values = 1 / (s ** 2)
    return vif_values

# Preprocessor Data , scaling, PCA, and feature selection
def preprocess_data(X_train, X_test, variance_threshold=0.97):
    # Normalize features
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Apply PCA
    pca = PCA(n_components=variance_threshold)  # Retain 97% variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Use original feature names for PCA components
    loadings = pd.DataFrame(pca.components_, columns=X_train.columns)

    # Create descriptive names for each principal component using all contributing features
    feature_names = ["_".join(loadings.iloc[i].index.tolist()) for i in range(loadings.shape[0])]

    # Assign human-readable column names
    X_train_pca = pd.DataFrame(X_train_pca, columns=feature_names)
    X_test_pca = pd.DataFrame(X_test_pca, columns=feature_names)

    return X_train_pca, X_test_pca, scaler, pca

# Check class imbalance
def check_class_imbalance(y_train, encoder):
    """Checks and visualizes class imbalance with actual activity names."""
    label_counts = pd.Series(y_train).value_counts()

    # Convert numeric labels back to original activity names
    activity_labels = encoder.inverse_transform(label_counts.index)

    # Define colors for the pie chart
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold', 'pink', 'purple']

    # Plot class distribution with activity names
    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=activity_labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title("Distribution of Activity Classes")
    st.pyplot(plt)

    return label_counts

# Handle class imbalance
def handle_class_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance."""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Tune hyperparameters
def tune_hyperparameters(X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV."""
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
    st.write(f"Best hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# RandomForest Model Training and Evaluation
def randomforest_model_training_and_evaluation(algorithm_name, X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest model."""
    st.write(f"\n--- Training and Evaluating {algorithm_name} Model ---")

    # Hyperparameter tuning
    best_model = tune_hyperparameters(X_train, y_train)

    # Initialize the model with the best hyperparameters
    model = RandomForestClassifier(
        n_estimators=best_model.n_estimators,
        max_depth=best_model.max_depth,
        min_samples_split=best_model.min_samples_split, random_state=42, class_weight="balanced"
    )
    st.write(f"{algorithm_name} model initialized with tuned hyperparameters.")

    # Perform cross-validation
    logging.info("Performing cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    st.write(f"Cross-validation accuracy scores: {cv_scores}")
    st.write(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")

    # Initialize the model
    st.write(f"{algorithm_name} model trained on the full training set.")

    # Train the model
    model.fit(X_train, y_train)
    st.write("Model trained.")

    # Make predictions
    y_pred_random_forest = model.predict(X_test)
    st.write("Predictions made on the test set.")

    # Evaluate model performance
    evaluate_model(algorithm_name, y_test, y_pred_random_forest)
    st.write("Model evaluated.")
    return model, y_pred_random_forest

# Plot confusion matrix
def plot_confusion_matrix(cm, y_test):
    """Plot and save the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(plt)

# Evaluate model
def evaluate_model(algorithm_name, y_test, y_pred):
    """Evaluate the model's performance on the test set."""
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print evaluation metrics
    st.write(f"Test Accuracy: {accuracy * 100:.2f}%")
    st.write(f"Test F1-score: {f1:.4f}")
    st.write(f"Test Precision: {precision:.4f}")
    st.write(f"Test Recall: {recall:.4f}")
    st.write("\nClassification Report:\n" + report)

    # Plot confusion matrix
    plot_confusion_matrix(cm, y_test)
    return accuracy, report, cm

# SVM Model Training and Evaluation
def svm_model_training_and_evaluation(X_train, y_train, X_test, y_test, algorithm_name="SVM"):
    """Train, evaluate, and save an SVM model."""
    st.write(f"Training {algorithm_name} model...")

    # Hyperparameter tuning
    st.write("Starting hyperparameter tuning...")
    best_model = svm_tune_hyperparameters(X_train, y_train)
    st.write("Hyperparameter tuning completed.")

    # Initialize the model
    model_svm = SVC(
        C=best_model.C,
        kernel=best_model.kernel,
        random_state=42,
        probability=True  # Enable probability estimates
    )
    st.write(f"{algorithm_name} model initialized with tuned hyperparameters.")

    # Perform cross-validation
    logging.info("Performing cross-validation...")
    cv_scores = cross_validate(model_svm, X_train, y_train)
    st.write(f"Cross-validation accuracy scores: {cv_scores}")
    st.write(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")

    # Train the model
    model_svm.fit(X_train, y_train)
    st.write("Model trained.")

    # Make predictions on the test set
    y_pred_svm = model_svm.predict(X_test)
    st.write("Predictions made on the test set.")

    # Evaluate the model
    evaluate_model(algorithm_name, y_test, y_pred_svm)
    st.write("Model evaluated.")

    return model_svm, y_pred_svm


# CNN Model Training and Evaluation
def cnn_model_training_and_evaluation(X_train, y_train, X_test, y_test):
    """Train and evaluate a CNN model."""
    # Reshape data for CNN input
    X_train_reshaped = np.expand_dims(X_train, axis=-1)  # Shape: (samples, timesteps, 1)
    X_test_reshaped = np.expand_dims(X_test, axis=-1)    # Shape: (samples, timesteps, 1)

    # Define input shape and number of classes
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])  # (timesteps, features)
    num_classes = len(np.unique(y_train))  # Number of unique classes in y_train
    st.write("Input shape:", input_shape)
    st.write("Number of classes:", num_classes)

    # Build the CNN model
    def build_cnn_model(input_shape, num_classes):
        """Build a CNN model."""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            Conv1D(256, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            Flatten(),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Build the model
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Make predictions
    y_pred_cnn = model.predict(X_test_reshaped).argmax(axis=1)

    # Classification report and confusion matrix
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred_cnn))

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_cnn))

    return model


# CNN-LSTM Hybrid Model Training and Evaluation
def hybrid_cnn_lstm_model_training_and_evaluation(X_train, y_train, X_test, y_test):
    """Train and evaluate a CNN-LSTM hybrid model."""
    # Reshape data for CNN input
    X_train_reshaped = np.expand_dims(X_train, axis=-1)  # Shape: (samples, timesteps, 1)
    X_test_reshaped = np.expand_dims(X_test, axis=-1)    # Shape: (samples, timesteps, 1)

    # Define input shape and number of classes
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])  # (timesteps, features)
    num_classes = len(np.unique(y_train))  # Number of unique classes in y_train
    st.write("Input shape:", input_shape)
    st.write("Number of classes:", num_classes)

    # Build the CNN-LSTM model
    def build_hybrid_model(input_shape, num_classes):
        """Build a CNN-LSTM hybrid model."""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),

            LSTM(64, return_sequences=True),
            Dropout(0.3),

            LSTM(32),
            Dropout(0.3),

            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Build the model
    model = build_hybrid_model(input_shape, num_classes)
    model.summary()

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    st.write(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Make predictions
    y_pred_hybrid = model.predict(X_test_reshaped).argmax(axis=1)

    # Classification report and confusion matrix
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred_hybrid))

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred_hybrid))

    return model

# Streamlit App
def main():
    st.title("Human Activity Recognition (HAR) Model Training and Evaluation")

    # Load data
    train, test = load_data(train_path, test_path)

    # Perform EDA
    if st.checkbox("Perform EDA"):
        EDA_check(train, test)

    # Preprocess data
    if st.checkbox("Preprocess Data"):
        X_train, y_train, X_test, y_test, scaler, pca, activity_encoder, subject_encoder = data_preprocessing(train, test)

    # Train and evaluate models
    if st.checkbox("Train and Evaluate RandomForest Model"):
        model_random_forest, y_pred_random_forest = randomforest_model_training_and_evaluation("RandomForest", X_train, y_train, X_test, y_test)

    if st.checkbox("Train and Evaluate SVM Model"):
        model_svm, y_pred_svm = svm_model_training_and_evaluation(X_train, y_train, X_test, y_test)

    if st.checkbox("Train and Evaluate CNN-LSTM Hybrid Model"):
        hybrid_model = hybrid_cnn_lstm_model_training_and_evaluation(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()

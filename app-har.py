import tensorflow as tf
from keras import Sequential
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import streamlit as st
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l2
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, make_scorer
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, BatchNormalization, Flatten, Input, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor


load_model = tf.keras.models.load_model
# Add these helper functions at the top of your code, right after the imports

def show_class_distribution_comparison(y_train, class_weights, encoder):
    """Show side-by-side comparison of original vs weighted distribution"""
    st.subheader("Class Distribution Comparison")

    # Get class names and counts
    classes, counts = np.unique(y_train, return_counts=True)
    class_names = encoder.inverse_transform(classes)

    # Calculate weighted counts
    weights = np.array([class_weights[cls] for cls in classes])
    weighted_counts = counts * weights

    # Normalize weighted counts for visualization
    weighted_counts = weighted_counts / weighted_counts.sum() * counts.sum()

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Original distribution
    sns.barplot(x=class_names, y=counts, ax=ax1)
    ax1.set_title("Original Class Distribution")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

    # Weighted distribution
    sns.barplot(x=class_names, y=weighted_counts, ax=ax2)
    ax2.set_title("Weighted Class Distribution")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    st.pyplot(fig)

def show_prediction_samples(y_true, y_pred, encoder, num_samples=5, offsets=[0]):
    """Display sample predictions at specified offsets"""
    st.subheader("Prediction Samples")

    # Get class names
    y_true_names = encoder.inverse_transform(y_true)
    y_pred_names = encoder.inverse_transform(y_pred)

    # Display samples at different offsets
    for offset in offsets:
        st.write(f"Offset {offset} to {offset + num_samples - 1}:")
        for i in range(offset, offset + num_samples):
            if i < len(y_true):
                st.write(f"Sample {i}: Predicted = {y_pred_names[i]}, Original = {y_true_names[i]}")
        st.write("")

def train_xgboost(X_train, y_train, X_test, y_test, class_weights=None, model_params=None, random_state=42):
    """
    Updated version of your original XGBoost training function
    that works with the unified interface while keeping all your original functionality
    """
    if model_params is None:
        model_params = {}

    # Step 1: Split training set into train/val for early stopping
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state
    )

    # Convert class weights to sample weights (handling both dict and None)
    if class_weights:
        if isinstance(class_weights, dict):
            sample_weights_main = np.array([class_weights[int(label)] for label in y_train_main])
        else:
            sample_weights_main = None
    else:
        sample_weights_main = None

    # Define model with parameters from model_params or defaults
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(np.unique(y_train)),
        learning_rate=model_params.get('learning_rate', 0.1),
        max_depth=model_params.get('max_depth', 6),
        n_estimators=model_params.get('n_estimators', 1000),
        subsample=model_params.get('subsample', 0.8),
        colsample_bytree=model_params.get('colsample_bytree', 0.8),
        random_state=random_state,
        use_label_encoder=False,
        verbosity=1,
        early_stopping_rounds=model_params.get('early_stopping_rounds', 10),
        eval_metric='mlogloss'
    )

    # Fit model with validation set
    model.fit(
        X_train_main,
        y_train_main,
        sample_weight=sample_weights_main,
        eval_set=[(X_val, y_val)],
        verbose=model_params.get('verbose', True)
    )

    # Predict and return
    y_pred = model.predict(X_test)
    return model, y_pred

# def train_cnn_lstm(X_train, y_train, X_test, y_test, class_weights, model_params=None):
#     """Train CNN-LSTM model with proper parameter handling"""
#     if model_params is None:
#         model_params = {}
#
#     # Reshape data for CNN-LSTM
#     X_train_3d = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#     X_test_3d = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#
#     num_classes = len(np.unique(y_train))
#
#     model = Sequential([
#         Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
#         LSTM(64, return_sequences=True),
#         LSTM(32),
#         Dense(num_classes, activation='softmax')
#     ])
#
#     model.compile(
#         optimizer=Adam(model_params.get('learning_rate', 0.001)),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#
#     history = model.fit(
#         X_train_3d, y_train,
#         validation_data=(X_test_3d, y_test),
#         epochs=model_params.get('epochs', 20),
#         batch_size=model_params.get('batch_size', 32),
#         class_weight=class_weights,
#         verbose=1
#     )
#
#     y_pred = model.predict(X_test_3d).argmax(axis=1)
#     return model, y_pred

# Set page config
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
train_path = "data/raw/train.csv"
test_path = "data/raw/test.csv"

# ===== CREATIVE HEADER SECTION =====
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #6e48aa 0%, #9d50bb 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.3);
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .sensor-animation {
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        width: 120px;
        opacity: 0.8;
    }
    .activity-tags {
        display: flex;
        gap: 0.8rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    .activity-tag {
        background: rgba(255,255,255,0.15);
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        color: white;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>

<div class="header-container">
    <div>
        <div class="header-title">HUMAN ACTIVITY RECOGNITION DASHBOARD</div>
        <div class="header-subtitle">Real-time motion pattern analysis from multi-sensor data</div>
        <div class="activity-tags">
            <div class="activity-tag">🏃 Walking</div>
            <div class="activity-tag">🛌 Lying</div>
            <div class="activity-tag">🧘 Sitting</div>
            <div class="activity-tag">🏋️ Standing</div>
            <div class="activity-tag">↗️ Upstairs</div>
            <div class="activity-tag">↘️ Downstairs</div>
        </div>
    </div>
    <svg class="sensor-animation" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="30" r="5" fill="#fff" opacity="0.8">
            <animate attributeName="cy" values="30;70;30" dur="3s" repeatCount="indefinite"/>
        </circle>
        <circle cx="30" cy="50" r="5" fill="#fff" opacity="0.6">
            <animate attributeName="cx" values="30;70;30" dur="4s" repeatCount="indefinite"/>
        </circle>
        <circle cx="70" cy="50" r="5" fill="#fff" opacity="0.7">
            <animate attributeName="r" values="5;8;5" dur="2s" repeatCount="indefinite"/>
        </circle>
    </svg>
</div>
""", unsafe_allow_html=True)

# Sensor visualization
with st.expander("📊 Live Sensor Feed Simulation", expanded=True):
    # First row - Full width activity detection
    st.markdown("**Current Activity Detection**")
    st.image(
        "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3kxdzhxazJnOHZqY3FpdjZ6OW55dHY4Y25jMWk4bTVyaWRqbGU1byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/duGB9Or2KTW4aB4KhY/giphy.gif",
        width=800  # Wider image
    )

    # Second row - Sensor data in columns
    st.markdown("**Sensor Data**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Accelerometer**")
        accel_data = pd.DataFrame({
            'X': np.random.normal(0, 0.5, 20),
            'Y': np.random.normal(0, 0.5, 20),
            'Z': np.random.normal(1, 0.3, 20)
        }, index=range(20))
        st.line_chart(accel_data)
    with col2:
        st.markdown("**Gyroscope**")
        gyro_data = pd.DataFrame({
            'X': np.random.normal(0, 0.2, 20),
            'Y': np.random.normal(0, 0.2, 20),
            'Z': np.random.normal(0, 0.1, 20)
        }, index=range(20))
        st.line_chart(gyro_data)
    with col3:
        st.markdown("**Additional Metrics**")
        # You can add another visualization or leave empty
        st.metric("Motion Intensity", "High", "2.5%")
        st.metric("Activity Confidence", "89%", "1.2%")

# Load data function
@st.cache_data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

# EDA functions
def perform_eda(train, test):
    st.subheader("Exploratory Data Analysis")

    # Basic info
    with st.expander("Dataset Overview"):
        st.write("Train Data Shape:", train.shape)
        st.write("Test Data Shape:", test.shape)
        st.write("Train Columns:", train.columns.tolist())

    # Class distribution
    with st.expander("Class Distribution"):
        fig, ax = plt.subplots()
        train['Activity'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    # Missing values
    with st.expander("Missing Values"):
        st.write("Train Missing Values:", train.isnull().sum().sum())
        st.write("Test Missing Values:", test.isnull().sum().sum())

    # Correlation matrix
    with st.expander("Feature Correlation"):
        numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = train[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, ax=ax)
        st.pyplot(fig)

# Preprocessing functions
def preprocess_data(train, test, preprocess_options):
    # Encode target
    le = LabelEncoder()
    y_train = le.fit_transform(train['Activity'])
    y_test = le.transform(test['Activity'])

    # Drop non-feature columns
    X_train = train.drop(['Activity', 'subject'], axis=1)
    X_test = test.drop(['Activity', 'subject'], axis=1)

    # Feature selection
    if 'remove_high_corr' in preprocess_options:
        X_train, X_test = remove_highly_correlated_features(X_train, X_test)

    if 'remove_high_vif' in preprocess_options:
        X_train, X_test, _ = remove_high_vif_features(X_train, X_test)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, scaler, le, X_train.columns.tolist()

def remove_highly_correlated_features(train, test, threshold=0.99):
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return train.drop(to_drop, axis=1), test.drop(to_drop, axis=1)

def remove_high_vif_features(train, test, threshold=100):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = train.columns
    vif_data["VIF"] = [variance_inflation_factor(train.values, i) for i in range(train.shape[1])]
    high_vif = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
    return train.drop(high_vif, axis=1), test.drop(high_vif, axis=1), vif_data

# Model training functions
def train_logistic_regression(X_train, y_train, X_test, y_test, class_weights):
    model = LogisticRegression(class_weight=class_weights, max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_random_forest(X_train, y_train, X_test, y_test, class_weights, use_smote=False):
    if use_smote:
        smote = SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)
        pipeline = ImbPipeline([
            ('smote', smote),
            ('rf', RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1))
        ])
    else:
        pipeline = Pipeline([
            ('rf', RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1))
        ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return pipeline, y_pred

def train_svm(X_train, y_train, X_test, y_test, class_weights):
    model = SVC(class_weight=class_weights, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# def train_xgboost(X_train, y_train, X_test, y_test, class_weights):
#     model = XGBClassifier(scale_pos_weight=class_weights, random_state=42, n_jobs=-1)
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     return model, y_pred

def train_cnn_lstm(X_train, y_train, X_test, y_test, class_weights):
    """
    Your original CNN-LSTM training code properly integrated into app.py
    Takes exactly 5 parameters to match the calling pattern
    """
    # Standardize the data (if not already scaled in preprocessing)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape data for Conv1D
    X_train_3d = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_3d = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build the model
    input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Input(shape=input_shape),
        # Conv Block 1
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        # Conv Block 2
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        # LSTM Layers
        LSTM(128, return_sequences=True, dropout=0.2),
        LSTM(64, dropout=0.2),
        # Dense Layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Compile with your original settings
    loss = SparseCategoricalCrossentropy(from_logits=False, reduction='sum_over_batch_size')
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])

    # Prepare callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    ]

    # Train the model
    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_test_3d, y_test),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Make predictions
    y_pred = model.predict(X_test_3d).argmax(axis=1)
    return model, y_pred



# Add these model training functions to your app.py
def train_improved_dnn(X_train, y_train, X_test, y_test, class_weights):
    """Train an improved DNN model with regularization"""
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Dense(512, activation='relu', input_shape=input_shape, kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    return model, y_pred

def train_cnn(X_train, y_train, X_test, y_test, class_weights):
    """Train a CNN model"""
    X_train_reshaped = np.expand_dims(X_train, axis=-1)
    X_test_reshaped = np.expand_dims(X_test, axis=-1)
    input_shape = (X_train_reshaped.shape[1], X_train_reshaped.shape[2])
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(256, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    model.fit(
        X_train_reshaped, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_reshaped, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = model.predict(X_test_reshaped).argmax(axis=1)
    return model, y_pred

def train_cnn_dense(X_train, y_train, X_test, y_test, class_weights):
    """Train a CNN + Dense hybrid model"""
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)
    input_shape = X_train_cnn.shape[1:]
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    ]

    model.fit(
        X_train_cnn, y_train,
        validation_data=(X_test_cnn, y_test),
        epochs=150,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
    return model, y_pred

class WeightedSMOTEPipeline(Pipeline):
    """Custom pipeline to handle sample weights with SMOTE"""
    def fit(self, X, y, **fit_params):
        rf_sample_weight = fit_params.pop('rf__sample_weight', None)
        X_res, y_res = self.named_steps['smote'].fit_resample(X, y)

        if rf_sample_weight is not None:
            synthetic_weights = np.ones(len(y_res) - len(y))
            fit_params['rf__sample_weight'] = np.concatenate([rf_sample_weight, synthetic_weights])

        super().fit(X_res, y_res, **fit_params)
        return self

def train_random_forest_with_smote(X_train, y_train, X_test, y_test, class_weights):
    """Train Random Forest with SMOTE oversampling"""
    train_sample_weights = np.array([class_weights[y] for y in y_train])

    pipeline = WeightedSMOTEPipeline([
        ('smote', SMOTE(sampling_strategy='not majority', k_neighbors=3, random_state=42)),
        ('rf', RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True, class_weight=None))
    ])

    param_grid = {
        'rf__n_estimators': [500],
        'rf__max_depth': [30],
        'rf__min_samples_split': [7]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5),
        scoring=make_scorer(balanced_accuracy_score),
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train, rf__sample_weight=train_sample_weights)
    y_pred = grid.best_estimator_.predict(X_test)
    return grid.best_estimator_, y_pred

# Evaluation function
def evaluate_model(y_true, y_pred, encoder, model_name):
    st.subheader(f"Evaluation Results for {model_name}")

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    # Display metrics
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{accuracy:.2%}")
    cols[1].metric("F1 Score", f"{f1:.4f}")
    cols[2].metric("Precision", f"{precision:.4f}")
    cols[3].metric("Recall", f"{recall:.4f}")

    # Classification report
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# Main app
def main():
    # Sidebar controls
    st.sidebar.header("DATA UPLOAD")

    # Data selection
    data_option = st.sidebar.radio("Data Source", ["Use sample data", "Upload your own"])

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["EDA", "Preprocessing", "Model Training"])

    # data loading section
    if data_option == "Use sample data":
        train, test = load_data(train_path, test_path)
    else:
        uploaded_train = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])

        # Make test data optional
        uploaded_test = st.sidebar.file_uploader("Upload Test Data (CSV) - Optional", type=["csv"])

        if uploaded_train:
            train = pd.read_csv(uploaded_train)

            if uploaded_test:  # User provided test data
                test = pd.read_csv(uploaded_test)
            else:  # Split training data
                if st.checkbox("Split training data for validation?"):
                    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)
                    train, test = train_test_split(train, test_size=test_size, random_state=42)
                else:
                    test = train.copy()  # Use training as test (not recommended but possible)
        else:
            st.warning("Please upload at least training data")
            return

    # Store data in session state
    st.session_state.train = train
    st.session_state.test = test

    # EDA Tab
    with tab1:
        perform_eda(train, test)

    # Preprocessing Tab
    with tab2:
        st.header("Data Preprocessing")

        # Create expandable sections for each preprocessing step
        with st.expander("Standard Scaling", expanded=True):
            do_scaling = st.checkbox("Apply Standard Scaling", value=True, key='scale')

        with st.expander("Feature Removal by Correlation"):
            remove_corr = st.checkbox("Remove Highly Correlated Features", value=False, key='corr')
            if remove_corr:
                corr_threshold = st.slider(
                    "Correlation Threshold",
                    0.7, 1.0, 0.95, 0.01,
                    key="corr_thresh"
                )

        with st.expander("Feature Removal by VIF"):
            remove_vif = st.checkbox("Remove High VIF Features", value=False, key='vif')
            if remove_vif:
                vif_threshold = st.slider(
                    "VIF Threshold",
                    50, 200, 100, 5,
                    key="vif_thresh"
                )

        # with st.expander("Feature Selection"):
        #     do_rfe = st.checkbox("Apply Feature Selection (RFE)", value=False, key='rfe')
        #     if do_rfe:
        #         rfe_features = st.slider(
        #             "Number of Features to Select",
        #             10, min(100, len(train.columns)-2), 50, 5,
        #             key="rfe_feats"
        #         )

        with st.expander("Class Imbalance Handling"):
            handle_imbalance = st.checkbox("Handle Class Imbalance", value=True, key='imb')
            if handle_imbalance:
                imbalance_method = st.radio(
                    "Method",
                    ["Class Weighting", "SMOTE Oversampling"],
                    index=0,
                    key="imb_method"
                )

        # Add a button to execute preprocessing
        if st.button("Run Preprocessing", type="primary", key='run_preprocess'):
            with st.spinner("Preprocessing data..."):
                try:
                    # Initialize variables
                    X_train = train.drop(['Activity', 'subject'], axis=1)
                    X_test = test.drop(['Activity', 'subject'], axis=1)
                    y_train = train['Activity']
                    y_test = test['Activity']

                    # Encode labels
                    le = LabelEncoder()
                    y_train_encoded = le.fit_transform(y_train)
                    y_test_encoded = le.transform(y_test)

                    # 1. Remove highly correlated features
                    if remove_corr:
                        X_train, X_test = remove_highly_correlated_features(
                            X_train, X_test, corr_threshold
                        )

                    # 2. Remove high VIF features
                    if remove_vif:
                        X_train, X_test, vif_report = remove_high_vif_features(
                            X_train, X_test, vif_threshold
                        )
                        st.session_state.vif_report = vif_report

                    # 3. Standard scaling
                    if do_scaling:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        st.session_state.scaler = scaler
                    else:
                        X_train_scaled = X_train.values
                        X_test_scaled = X_test.values

                    # 4. Feature selection (RFE)
                    # if do_rfe:
                    #     rfe_selector = RFE(
                    #         estimator=RandomForestClassifier(n_estimators=100),
                    #         n_features_to_select=rfe_features
                    #     )
                    #     X_train_scaled = rfe_selector.fit_transform(X_train_scaled, y_train_encoded)
                    #     X_test_scaled = rfe_selector.transform(X_test_scaled)
                    #     st.session_state.selected_features = rfe_selector.get_support()

                    # Store processed data in session state
                    st.session_state.update({
                        'X_train': X_train_scaled,
                        'X_test': X_test_scaled,
                        'y_train': y_train_encoded,
                        'y_test': y_test_encoded,
                        'encoder': le,
                        'preprocessing_complete': True,
                        'feature_names': X_train.columns.tolist()
                    })

                    # Compute class weights
                    class_weights = compute_class_weight(
                        class_weight="balanced",
                        classes=np.unique(y_train_encoded),
                        y=y_train_encoded
                    )
                    st.session_state.class_weights = dict(zip(np.unique(y_train_encoded), class_weights))

                    # After computing class weights in preprocessing:
                    # # Show comparison
                    # show_class_distribution_comparison(
                    #     y_train_encoded,
                    #     st.session_state.class_weights,
                    #     le
                    # )

                    st.success("Preprocessing completed successfully!")

                    # Show preprocessing summary and class weight distribution
                    with st.expander("Preprocessing Summary", expanded=True):
                        col1, col2 = st.columns(2)
                        col1.metric("Final Training Shape", f"{X_train_scaled.shape}")
                        col2.metric("Final Test Shape", f"{X_test_scaled.shape}")

                        if remove_vif and 'vif_report' in st.session_state:
                            st.write("VIF Report:")
                            st.dataframe(st.session_state.vif_report)

                        # Show class weight distribution
                        # Show comparison
                        show_class_distribution_comparison(
                            y_train_encoded,
                            st.session_state.class_weights,
                            le
                        )

                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")

    # Model Training Tab
    with tab3:
        st.header("Model Training")

        if 'preprocessing_complete' not in st.session_state:
            st.warning("Please complete preprocessing first")
        else:
            # Show class weight distribution if available
            # if 'class_weights' in st.session_state:
            #     show_class_distribution_comparison(
            #         y_train_encoded,
            #         st.session_state.class_weights,
            #         le
            #     )

            model_option = st.selectbox(
                "Select Model",
                [
                    "Logistic Regression",
                    "Random Forest",
                    "SVM",
                    "XGBoost",
                    "CNN-LSTM Hybrid",
                    "Improved DNN",
                    "CNN",
                    "CNN-Dense Hybrid"#,
                  #  "Random Forest with SMOTE"
                ],
                key='model_select'
            )

            # Model-specific parameters
            model_params = {}
            if model_option == "Random Forest":
                model_params['n_estimators'] = st.slider("Number of Trees", 50, 500, 100, key='rf_trees')
                model_params['max_depth'] = st.slider("Max Depth", 3, 20, 10, key='rf_depth')

            if model_option == "XGBoost":
                model_params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key='xgb_lr')
                model_params['max_depth'] = st.slider("Max Depth", 3, 15, 6, key='xgb_depth')

            if model_option == "CNN-LSTM Hybrid":
                model_params['epochs'] = st.slider("Epochs", 10, 100, 20, key='cnn_epochs')
                model_params['batch_size'] = st.selectbox("Batch Size", [16, 32, 64, 128], index=2, key='cnn_batch')
                model_params['learning_rate'] = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, key='cnn_lr')

            if st.button("Train Model", type="primary", key='train_model'):
                with st.spinner(f"Training {model_option}..."):
                    try:
                        if model_option == "Logistic Regression":
                            model, y_pred = train_logistic_regression(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                        elif model_option == "Random Forest":
                            model, y_pred = train_random_forest(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights,
                                model_params
                            )
                        elif model_option == "SVM":
                            model, y_pred = train_svm(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                        elif model_option == "XGBoost":
                            model, y_pred = train_xgboost(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                            # Calculate test accuracy if needed
                          #  test_acc = accuracy_score(st.session_state.y_test, y_pred)

                        elif model_option == "CNN-LSTM Hybrid":
                            model, y_pred = train_cnn_lstm(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                        elif model_option == "Improved DNN":
                            model, y_pred = train_improved_dnn(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                        elif model_option == "CNN":
                            model, y_pred = train_cnn(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                        elif model_option == "CNN-Dense Hybrid":
                            model, y_pred = train_cnn_dense(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )
                        elif model_option == "Random Forest with SMOTE":
                            model, y_pred = train_random_forest_with_smote(
                                st.session_state.X_train,
                                st.session_state.y_train,
                                st.session_state.X_test,
                                st.session_state.y_test,
                                st.session_state.class_weights
                            )

                        # Store results
                        st.session_state.model = model
                        st.session_state.y_pred = y_pred
                        st.session_state.current_model = model_option

                        st.success("Model training complete!")
                        evaluate_model(
                            st.session_state.y_test,
                            st.session_state.y_pred,
                            st.session_state.encoder,
                            model_option
                        )

                        # Show prediction samples
                        show_prediction_samples(
                            st.session_state.y_test,
                            st.session_state.y_pred,
                            st.session_state.encoder
                        )

                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")

            # Show feature importance if available
            if 'model' in st.session_state and st.session_state.current_model == model_option:
                if hasattr(st.session_state.model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    if isinstance(st.session_state.model, Pipeline):
                        importances = st.session_state.model.named_steps['rf'].feature_importances_
                    else:
                        importances = st.session_state.model.feature_importances_

                    feat_imp = pd.DataFrame({
                        'Feature': st.session_state.feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='Importance', y='Feature', data=feat_imp.head(20), ax=ax)
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
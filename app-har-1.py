import tensorflow as tf
from keras import Sequential
import tensorflow as tf
from keras.src.callbacks import ReduceLROnPlateau
from sklearn.utils import compute_class_weight
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

import hashlib
import glob
from datetime import datetime

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
import hashlib
import glob
from datetime import datetime

# ===========================================
# Helper Functions
# ===========================================

def get_data_hash(train, test, options=None):
    """Generate unique hash for data and options"""
    hash_obj = hashlib.md5()
    hash_obj.update(pd.util.hash_pandas_object(train).values.tobytes())
    hash_obj.update(pd.util.hash_pandas_object(test).values.tobytes())
    if options:
        hash_obj.update(str(options).encode())
    return hash_obj.hexdigest()

def save_to_cache(data, cache_dir, cache_name):
    """Save data to cache directory"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.pkl")
    joblib.dump(data, cache_path)
    return cache_path

def load_from_cache(cache_dir, cache_name):
    """Load data from cache if exists"""
    cache_path = os.path.join(cache_dir, f"{cache_name}.pkl")
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    return None

def cleanup_old_files(directory, max_files=5):
    """Keep only the most recent files"""
    try:
        files = sorted(glob.glob(os.path.join(directory, "*")),
                       key=os.path.getmtime, reverse=True)
        for old_file in files[max_files:]:
            os.remove(old_file)
    except Exception as e:
        st.warning(f"Could not clean up old files: {str(e)}")

def show_class_distribution_comparison(y_train, class_weights, encoder):
    """Show side-by-side comparison of original vs weighted distribution"""
    st.subheader("Class Distribution Comparison")
    classes, counts = np.unique(y_train, return_counts=True)
    class_names = encoder.inverse_transform(classes)
    weights = np.array([class_weights[cls] for cls in classes])
    weighted_counts = counts * weights
    weighted_counts = weighted_counts / weighted_counts.sum() * counts.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=class_names, y=counts, ax=ax1)
    ax1.set_title("Original Class Distribution")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    sns.barplot(x=class_names, y=weighted_counts, ax=ax2)
    ax2.set_title("Weighted Class Distribution")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig)

def show_prediction_samples(y_true, y_pred, encoder, num_samples=5, offsets=[0]):
    """Display sample predictions"""
    st.subheader("Prediction Samples")
    y_true_names = encoder.inverse_transform(y_true)
    y_pred_names = encoder.inverse_transform(y_pred)

    for offset in offsets:
        st.write(f"Offset {offset} to {offset + num_samples - 1}:")
        for i in range(offset, offset + num_samples):
            if i < len(y_true):
                st.write(f"Sample {i}: Predicted = {y_pred_names[i]}, Original = {y_true_names[i]}")
        st.write("")

# ===========================================
# Data Loading with Caching
# ===========================================

@st.cache_data
def load_data(train_path, test_path):
    """Load data with caching"""
    cache_dir = "cache/data"
    cache_name = f"data_{os.path.basename(train_path)}_{os.path.basename(test_path)}"
    cached_data = load_from_cache(cache_dir, cache_name)

    if cached_data:
        st.success("Loaded data from cache!")
        return cached_data['train'], cached_data['test']

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    save_to_cache({'train': train, 'test': test}, cache_dir, cache_name)
    return train, test

# ===========================================
# EDA with Caching
# ===========================================

def perform_eda(train, test):
    """Perform EDA with caching"""
    cache_dir = "cache/eda"
    cache_name = f"eda_{get_data_hash(train, test)}"
    cached_eda = load_from_cache(cache_dir, cache_name)

    if cached_eda:
        st.success("Loaded EDA from cache!")
        # Display cached results
        with st.expander("Dataset Overview (Cached)"):
            st.write("Train Data Shape:", cached_eda['shape']['train'])
            st.write("Test Data Shape:", cached_eda['shape']['test'])
            st.write("Train Columns:", cached_eda['shape']['columns'])

        with st.expander("Class Distribution (Cached)", expanded=False):
            st.pyplot(cached_eda['class_dist'])

        with st.expander("Missing Values (Cached)", expanded=False):
            st.write("Train Missing Values:", cached_eda['missing']['train'])
            st.write("Test Missing Values:", cached_eda['missing']['test'])

        with st.expander("Feature Correlation (Cached)", expanded=False):
            st.pyplot(cached_eda['correlation'])
        return

    st.subheader("Exploratory Data Analysis")
    eda_results = {}

    # Basic info
    with st.expander("Dataset Overview"):
        eda_results['shape'] = {
            'train': train.shape,
            'test': test.shape,
            'columns': train.columns.tolist()
        }
        st.write("Train Data Shape:", train.shape)
        st.write("Test Data Shape:", test.shape)
        st.write("Train Columns:", train.columns.tolist())

    # Class distribution
    with st.expander("Class Distribution"):
        fig, ax = plt.subplots()
        train['Activity'].value_counts().plot(kind='bar', ax=ax)
        eda_results['class_dist'] = fig
        st.pyplot(fig)

    # Missing values
    with st.expander("Missing Values"):
        eda_results['missing'] = {
            'train': train.isnull().sum().sum(),
            'test': test.isnull().sum().sum()
        }
        st.write("Train Missing Values:", train.isnull().sum().sum())
        st.write("Test Missing Values:", test.isnull().sum().sum())

    # Correlation matrix
    with st.expander("Feature Correlation"):
        numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = train[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, ax=ax)
        eda_results['correlation'] = fig
        st.pyplot(fig)

    save_to_cache(eda_results, cache_dir, cache_name)
    cleanup_old_files(cache_dir)

# ===========================================
# Preprocessing with Caching
# ===========================================

def remove_highly_correlated_features(train, test, threshold=0.99):
    """Remove highly correlated features"""
    corr_matrix = train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return train.drop(to_drop, axis=1), test.drop(to_drop, axis=1)

def remove_high_vif_features(train, test, threshold=100):
    """Remove high VIF features"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = train.columns
    vif_data["VIF"] = [variance_inflation_factor(train.values, i) for i in range(train.shape[1])]
    high_vif = vif_data[vif_data['VIF'] > threshold]['Feature'].tolist()
    return train.drop(high_vif, axis=1), test.drop(high_vif, axis=1), vif_data

def preprocess_data(train, test, preprocess_options, force_redo=False):
    """Preprocess data with caching"""
    cache_dir = "cache/preprocessing"
    cache_name = f"preprocessed_{get_data_hash(train, test, preprocess_options)}"

    if not force_redo:
        cached_data = load_from_cache(cache_dir, cache_name)
        if cached_data:
            st.success("Loaded preprocessed data from cache!")
            # Update session state
            st.session_state.update({
                'X_train': cached_data['X_train'],
                'X_test': cached_data['X_test'],
                'y_train': cached_data['y_train'],
                'y_test': cached_data['y_test'],
                'encoder': cached_data['encoder'],
                'preprocessing_complete': True,
                'feature_names': cached_data['feature_names'],
                'class_weights': cached_data['class_weights'],
                'scaler': cached_data['scaler']
            })
            return cached_data

    with st.spinner("Preprocessing data (this may take a while)..."):
        try:
            # Original preprocessing logic
            le = LabelEncoder()
            y_train = le.fit_transform(train['Activity'])
            y_test = le.transform(test['Activity'])

            X_train = train.drop(['Activity', 'subject'], axis=1)
            X_test = test.drop(['Activity', 'subject'], axis=1)

            # Feature selection
            if preprocess_options.get('remove_high_corr', False):
                X_train, X_test = remove_highly_correlated_features(
                    X_train, X_test, preprocess_options.get('corr_threshold', 0.95)
                )

            if preprocess_options.get('remove_high_vif', False):
                X_train, X_test, vif_report = remove_high_vif_features(
                    X_train, X_test, preprocess_options.get('vif_threshold', 100)
                )
                st.session_state.vif_report = vif_report

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Compute class weights
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train
            )
            class_weights = dict(zip(np.unique(y_train), class_weights))

            # Prepare results
            preprocessed = {
                'X_train': X_train_scaled,
                'X_test': X_test_scaled,
                'y_train': y_train,
                'y_test': y_test,
                'encoder': le,
                'feature_names': X_train.columns.tolist(),
                'class_weights': class_weights,
                'scaler': scaler
            }

            # Save to cache
            save_to_cache(preprocessed, cache_dir, cache_name)
            cleanup_old_files(cache_dir)

            # Update session state
            st.session_state.update({
                **preprocessed,
                'preprocessing_complete': True
            })

            st.success("Preprocessing completed successfully!")
            return preprocessed

        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")
            return None

# ===========================================
# Model Training with Caching
# ===========================================

def train_logistic_regression(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train logistic regression model"""
    model = LogisticRegression(
        class_weight=class_weights,
        max_iter=1000,
        n_jobs=-1,
        **model_params if model_params else {}
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_random_forest(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train random forest model"""
    model = RandomForestClassifier(
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1,
        **model_params if model_params else {}
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_svm(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train SVM model"""
    model = SVC(
        class_weight=class_weights,
        probability=True,
        random_state=42,
        **model_params if model_params else {}
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

def train_xgboost(X_train, y_train, X_test, y_test, class_weights, model_params=None, random_state=42):
    """Train XGBoost model"""
    if model_params is None:
        model_params = {}

    # Split for early stopping
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )

    # Sample weights
    sample_weights_main = None
    if class_weights:
        if isinstance(class_weights, dict):
            sample_weights_main = np.array([class_weights[int(label)] for label in y_train_main])

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

    model.fit(
        X_train_main,
        y_train_main,
        sample_weight=sample_weights_main,
        eval_set=[(X_val, y_val)],
        verbose=model_params.get('verbose', True)
    )
    y_pred = model.predict(X_test)
    return model, y_pred

def train_cnn_lstm(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train CNN-LSTM model"""
    # Standardize and reshape
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_3d = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_3d = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build model
    input_shape = (X_train_3d.shape[1], X_train_3d.shape[2])
    num_classes = len(np.unique(y_train))

    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.2),
        LSTM(128, return_sequences=True, dropout=0.2),
        LSTM(64, dropout=0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=model_params.get('learning_rate', 0.001) if model_params else Adam(learning_rate=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy']
                       ))

    # Callbacks
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        X_train_3d, y_train,
        validation_data=(X_test_3d, y_test),
        epochs=model_params.get('epochs', 100) if model_params else 100,
        batch_size=model_params.get('batch_size', 64) if model_params else 64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    y_pred = model.predict(X_test_3d).argmax(axis=1)
    return model, y_pred

def train_improved_dnn(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train improved DNN model"""
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
        optimizer=Adam(learning_rate=model_params.get('learning_rate', 0.0003) if model_params else Adam(learning_rate=0.0003),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy']
                       ))

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=4, min_lr=1e-6)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=model_params.get('epochs', 100) if model_params else 100,
        batch_size=model_params.get('batch_size', 64) if model_params else 64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    return model, y_pred

def train_cnn(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train CNN model"""
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

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]

    model.fit(
        X_train_reshaped, y_train,
        epochs=model_params.get('epochs', 50) if model_params else 50,
        batch_size=model_params.get('batch_size', 32) if model_params else 32,
        validation_data=(X_test_reshaped, y_test),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = model.predict(X_test_reshaped).argmax(axis=1)
    return model, y_pred

def train_cnn_dense(X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Train CNN-Dense hybrid model"""
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
        optimizer=Adam(learning_rate=model_params.get('learning_rate', 0.001) if model_params else Adam(learning_rate=0.001)),
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
        epochs=model_params.get('epochs', 150) if model_params else 150,
        batch_size=model_params.get('batch_size', 64) if model_params else 64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )

    y_pred = np.argmax(model.predict(X_test_cnn), axis=1)
    return model, y_pred

def load_or_train_model(model_name, train_func, X_train, y_train, X_test, y_test, class_weights, model_params=None):
    """Load or train model with caching"""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Create unique model ID based on data and parameters
    model_id = hashlib.md5()
    model_id.update(X_train.tobytes())
    model_id.update(y_train.tobytes())
    model_id.update(str(model_params).encode())
    model_id = model_id.hexdigest()

    model_path = os.path.join(models_dir, f"{model_name.lower().replace(' ', '_')}_{model_id}")

    # For TensorFlow models
    if model_name.lower() in ['cnn-lstm hybrid', 'improved dnn', 'cnn', 'cnn-dense hybrid']:
        model_path += ".h5"
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                if 'cnn' in model_name.lower():
                    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    y_pred = model.predict(X_test_reshaped).argmax(axis=1)
                else:
                    y_pred = np.argmax(model.predict(X_test), axis=1)
                st.success(f"Loaded pre-trained {model_name} from {model_path}")
                return model, y_pred
            except Exception as e:
                st.warning(f"Failed to load model: {str(e)}. Training new model...")

        model, y_pred = train_func(X_train, y_train, X_test, y_test, class_weights, model_params)
        model.save(model_path)
        st.success(f"Saved {model_name} to {model_path}")
        return model, y_pred

    # For scikit-learn style models
    else:
        model_path += ".pkl"
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                y_pred = model.predict(X_test)
                st.success(f"Loaded pre-trained {model_name} from {model_path}")
                return model, y_pred
            except Exception as e:
                st.warning(f"Failed to load model: {str(e)}. Training new model...")

        model, y_pred = train_func(X_train, y_train, X_test, y_test, class_weights, model_params)
        joblib.dump(model, model_path)
        st.success(f"Saved {model_name} to {model_path}")
        return model, y_pred

def evaluate_model(y_true, y_pred, encoder, model_name):
    """Evaluate model performance"""
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

# ===========================================
# Main App
# ===========================================

def main():
    # Initialize session state
    if 'force_redo' not in st.session_state:
        st.session_state.force_redo = False

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

    # Header section
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
    </div>
    """, unsafe_allow_html=True)

    # Sensor visualization
    with st.expander("📊 Live Sensor Feed Simulation", expanded=True):
        st.markdown("**Current Activity Detection**")
        st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3kxdzhxazJnOHZqY3FpdjZ6OW55dHY4Y25jMWk4bTVyaWRqbGU1byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/duGB9Or2KTW4aB4KhY/giphy.gif", width=800)

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
            st.metric("Motion Intensity", "High", "2.5%")
            st.metric("Activity Confidence", "89%", "1.2%")

    # Sidebar controls
    st.sidebar.header("DATA UPLOAD")
    data_option = st.sidebar.radio("Data Source", ["Use sample data", "Upload your own"])

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["EDA", "Preprocessing", "Model Training"])

    # Data loading section
    if data_option == "Use sample data":
        train, test = load_data(train_path, test_path)
    else:
        uploaded_train = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])
        uploaded_test = st.sidebar.file_uploader("Upload Test Data (CSV) - Optional", type=["csv"])

        if uploaded_train:
            train = pd.read_csv(uploaded_train)
            if uploaded_test:
                test = pd.read_csv(uploaded_test)
            else:
                if st.checkbox("Split training data for validation?"):
                    test_size = st.slider("Test size ratio", 0.1, 0.5, 0.2)
                    train, test = train_test_split(train, test_size=test_size, random_state=42)
                else:
                    test = train.copy()
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

        # Advanced options
        with st.expander("Advanced Options"):
            st.session_state.force_redo = st.checkbox(
                "Force reprocessing (ignore cache)",
                value=False,
                key='force_reprocess'
            )

        # Preprocessing options
        with st.expander("Standard Scaling", expanded=True):
            do_scaling = st.checkbox("Apply Standard Scaling", value=True, key='scale')

        with st.expander("Feature Removal by Correlation"):
            remove_corr = st.checkbox("Remove Highly Correlated Features", value=False, key='corr')
            if remove_corr:
                corr_threshold = st.slider(
                    "Correlation Threshold", 0.7, 1.0, 0.95, 0.01, key="corr_thresh")

        with st.expander("Feature Removal by VIF"):
            remove_vif = st.checkbox("Remove High VIF Features", value=False, key='vif')
            if remove_vif:
                vif_threshold = st.slider(
                    "VIF Threshold", 50, 200, 100, 5, key="vif_thresh")

        with st.expander("Class Imbalance Handling"):
            handle_imbalance = st.checkbox("Handle Class Imbalance", value=True, key='imb')
            if handle_imbalance:
                imbalance_method = st.radio(
                    "Method", ["Class Weighting", "SMOTE Oversampling"], index=0, key="imb_method")

        # Preprocessing button
        if st.button("Run Preprocessing", type="primary", key='run_preprocess'):
            preprocess_options = {
                'remove_high_corr': remove_corr,
                'corr_threshold': corr_threshold if remove_corr else None,
                'remove_high_vif': remove_vif,
                'vif_threshold': vif_threshold if remove_vif else None,
                'do_scaling': do_scaling}


if __name__ == "__main__":
    main()
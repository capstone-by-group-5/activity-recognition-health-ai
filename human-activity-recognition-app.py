
# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier


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


# Sensor visualization
with st.expander("📊 Live Sensor Feed Simulation", expanded=True):
    # First row - Full width activity detection
    st.markdown("**Human Activity Detection**")
    st.markdown(
        """
        <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExa3kxdzhxazJnOHZqY3FpdjZ6OW55dHY4Y25jMWk4bTVyaWRqbGU1byZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/duGB9Or2KTW4aB4KhY/giphy.gif" 
             width="1500" height="400">
        """,
        unsafe_allow_html=True
    )

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
        <div class="header-title">🏃‍♂️ HUMAN ACTIVITY RECOGNITION DASHBOARD</div>
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
        st.markdown("""
            <div style="width: 600px;">
                <h6>Motion Intensity</h6>
                <p><b>High</b> <span style="color: green;">(+2.5%)</span></p>
            </div>
            <div style="width: 300px;">
                <h6>Activity Confidence</h6>
                <p><b>89%</b> <span style="color: green;">(+1.2%)</span></p>
            </div>
        """, unsafe_allow_html=True)
        # You can add another visualization or leave empty
    # st.metric("Motion Intensity", "High", "2.5%")
    #st.metric("Activity Confidence", "89%", "1.2%")

# Create a Keras model
def create_model(neurons=64, activation='relu', dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(102,), activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6, activation='softmax'))  # Multi-class classification
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Load transformers and selected columns
@st.cache_resource
def load_transformers():
    scaler = joblib.load('models/scaler.pkl')
    pca = joblib.load('models/pca_transformer.pkl')
    encoder = joblib.load('models/label_encoder.pkl')
    return scaler, pca, encoder


# Load pre-trained models
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")


# Define class names for HAR dataset
class_names = ['STANDING', 'SITTING', 'LAYING', 'WALKING', 'WALKING_DOWNSTAIRS', 'WALKING_UPSTAIRS']

# Model selection options
model_dict = {
    "Logistic Regression": "logistic_model",
    "SVC": "svc_rbf_model",
    "XGBoost": "xgbmodel",
    "Artificial Neural Network (ANN)": "ann_model",
    "Random Forest": "randomforestmodel",
    "Ensemble Model (All of the above)": "ensemble_model"
}

# Streamlit UI
st.title("Model Evaluation")
st.write("Select a model and upload a test CSV to evaluate the accuracy.")

# Model selection using a radio button (single model selection)
selected_model = st.radio(
    "🔎 **Select a model to use:**",
    list(model_dict.keys())
)

# File uploader for test dataset
uploaded_file = st.file_uploader("📤 Upload a test CSV file", type=["csv"])

# Load transformers and selected columns
scaler, pca, encoder = load_transformers()

# Display instructions if no file uploaded
if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("✅ Test data successfully loaded!")

    # Check if 'Activity' column exists in test data
    if 'Activity' not in test_data.columns:
        st.error("❗️ The CSV must contain an 'Activity' column.")
    else:
        # Drop correlated columns
        X_test = test_data.drop(columns=['Activity'])
        y_test = test_data['Activity']
        y_test = encoder.transform(y_test)

        # Apply Standard Scaling
        X_scaled = scaler.transform(X_test)

        # Apply PCA on Test Data
        X_pca = pca.transform(X_scaled)

        # Load selected model
        model_file = model_dict[selected_model]
        model = load_model(model_file)

        # Predict and evaluate
        y_pred = model.predict(X_pca)
        accuracy = accuracy_score(y_test, y_pred)

        # Display results
        st.subheader(f"📊 Results for {selected_model}")
        st.write(f"🎯 **Accuracy:** {accuracy * 100:.2f}%")

        # Show prediction comparison
        st.write("🔎 **Predictions vs. Actual:**")
        results_df = pd.DataFrame({"Actual": encoder.inverse_transform(y_test), "Predicted": encoder.inverse_transform(y_pred)})
        st.write(results_df.sample(10))

else:
    st.warning("⚠️ Please upload a test CSV file to continue.")

# Footer
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit | Human Activity Recognition (HAR) Model")
# Human Activity Recognition for Health Monitoring Using Wearable Devices  

## 🚀 Key Features
- 7+ ML/DL Models (Logistic Regression, Random Forest, XGBoost, CNN, LSTM, Hybrid Models)
- Interactive Streamlit Dashboard
- Comprehensive Performance Metrics
- End-to-End Pipeline

## 📂 Project Structure

## 📌 Overview  
This project focuses on **Human Activity Recognition (HAR)** for **health monitoring** using wearable devices. It leverages **machine learning** and **deep learning** techniques to classify various human activities based on sensor data.  

## 🚀 Features  
- Data collection from wearable devices (accelerometer, gyroscope, etc.)  
- Preprocessing and feature engineering  
- Model training using machine learning/deep learning  
- Activity classification and real-time predictions  
- Potential applications in **health monitoring and fitness tracking**  

## 📂 Project Structure  

 ├── data/                   #Raw & processed data </br>
 ├── notebooks/              # Jupyter notebooks for analysis </br>
 ├── models/                 # Trained models </br>
 ├── src/                    # Source code </br>
 │ ├── preprocessing.py      # Data preprocessing </br>
 │ ├── training.py           # Model training </br>
 │ ├── inference.py          # Activity recognition </br>
 ├── results/                # Model evaluation results </br>
 ├── config/                 # Configuration files (Optional) </br>
 ├── scripts/                # Utility scripts (Optional) </br>
 ├── logs/                   # Training and evaluation logs (Optional) </br>
 ├── README.md # Project documentation


📂 Project Structure
├── data/
│ ├── raw/
│ │ ├── train.csv
│ │ ├── test.csv
│ ├── processed/
│ │ ├── train_processed.csv
│ │ ├── test_processed.csv
├── notebooks/
│ ├── eda.ipynb
├── models/
│ ├── har_model.pkl
│ ├── scaler.pkl
│ ├── pca.pkl
│ ├── encoder.pkl
├── src/
│ ├── preprocessing.py
│ ├── training.py
│ ├── inference.py
├── results/
│ ├── accuracy.txt
│ ├── classification_report.txt
│ ├── confusion_matrix.png
├── config/ (Optional)
├── scripts/ (Optional)
├── logs/ (Optional)
├── README.md


## 📥 Dataset
Dataset from Kaggle:  
[Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)  


## To Run the application
Command to Execute in terminal :  `python human-activity-recognition-app.py`

## Deployment  
[Human Activity Recognition with Smartphones - Deployment](https://activity-recognition-health-ai.streamlit.app/))  



## 🛠️ Setup & Execution
```bash
# 1. Clone repository  
git clone https://github.com/capstone-by-group-5/activity-recognition-health-ai.git  
cd activity-recognition-health-ai 

# 2. Create virtual environment  
python -m venv venv  
source venv/bin/activate       # Linux/Mac  
.\venv\Scripts\activate        # Windows  

# 3. Install dependencies  
pip install -r requirements.txt  

# 4. Download dataset (place in data/raw/)
# Get from: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones

# 5. Run Streamlit app  
streamlit run human-activity-recognition-app.py  


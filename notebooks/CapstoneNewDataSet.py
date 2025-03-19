#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install tensorflow-macos
#!pip install tensorflow-metal


# In[2]:


# Install PyTorch with MPS (Metal)
#!pip install torch torchvision torchaudio


# In[35]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# In[36]:


import zipfile
import os

# Path to your zip file
zip_file_path = 'Raw_time_domian_data.zip'

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('extracted_dataset')  # Extracts to 'extracted_dataset' folder

# Path to the extracted directory
extracted_path = 'extracted_dataset'


# In[37]:


# Column names to assign
column_names = [
    "time_acc", "acc_x", "acc_y", "acc_z",
    "time_gyro", "gyro_x", "gyro_y", "gyro_z"
]


# In[38]:


#!pip install polars


# In[39]:


import polars as pl
import os

# Path to parent directory with extracted data
parent_folder_path = 'extracted_dataset/1.Raw_time_domian_data'

# List to store DataFrames
dataframes = []

# Loop through all folders and CSVs
for activity_folder in os.listdir(parent_folder_path):
    activity_path = os.path.join(parent_folder_path, activity_folder)

    # Check if it's a directory (activity folder)
    if os.path.isdir(activity_path):
        print(f"📚 Loading data for activity: {activity_folder}")

        # Process each CSV in the activity folder
        for file in os.listdir(activity_path):
            if file.endswith('.csv'):
                file_path = os.path.join(activity_path, file)

                try:
                    # Read CSV with custom columns and streaming
                    df = pl.read_csv(file_path, has_header=False, new_columns=column_names)

                    # Add activity and file info for reference
                    df = df.with_columns([
                        pl.lit(activity_folder).alias('activity'),
                        pl.lit(file).alias('source_file')
                    ])

                    # Append DataFrame to list
                    dataframes.append(df)

                except Exception as e:
                    print(f"⚠️ Error loading {file}: {e}")

# Concatenate all DataFrames if not empty
if dataframes:
    combined_df = pl.concat(dataframes, how="vertical_relaxed", rechunk=True)
    print('✅ All CSVs loaded successfully!')
    print(combined_df.head())
else:
    print('❗ No valid CSV files found!')


# In[40]:


combined_df.shape


# In[41]:


# Remove duplicates based on all columns
combined_df = combined_df.unique()


# In[42]:


combined_df.shape


# In[43]:


# Check for rows with any null values
rows_with_nulls = combined_df.filter(combined_df.select(pl.all().is_null()).sum_horizontal() > 0)
print(rows_with_nulls)


# In[44]:


# Drop rows with any null values
clean_df = combined_df.drop_nulls()
clean_df.shape


# In[45]:


print(clean_df['activity'].unique())


# In[46]:


# Remove numbers followed by a dot
clean_df = clean_df.with_columns(
    pl.col("activity").str.replace(r"\d+\.\s*", "").alias("activity")
)


# In[47]:


clean_df.shape


# In[48]:


# Count class occurrences
class_counts = clean_df.group_by("activity").agg(
    pl.col("activity").count().alias("count")
)

print(class_counts)


# In[49]:


# Drop rows where 'activity' equals 'walking'
df_filtered = clean_df.filter(clean_df["activity"] != "Table-tennis")


# In[50]:


df_filtered.shape


# In[51]:


# Convert to Pandas for plotting
class_counts_pd = class_counts.to_pandas()

# Extract labels and values for plotting
labels = class_counts_pd["activity"]
counts = class_counts_pd["count"]


# In[52]:


# Plot the class distribution as a pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.Paired.colors
)
plt.title("Class Imbalance in Target Column")
plt.show()


# In[53]:


# Count occurrences of each activity
class_counts = df_filtered.group_by("activity").agg(
    pl.col("activity").count().alias("count")
)

# Convert to Pandas for plotting
class_counts_pd = class_counts.to_pandas()

# Extract labels and counts
labels = class_counts_pd["activity"]
counts = class_counts_pd["count"]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color="skyblue", edgecolor="black")

# Rotate and align x-axis labels
plt.xlabel("Activity")
plt.ylabel("Count")
plt.title("Activity Count Distribution")

# Rotate labels and align them to the right
plt.xticks(rotation=45, ha="right")

# Automatically adjust subplot to fit the labels
plt.tight_layout()

plt.show()


# In[54]:


# Define the number of records to sample per class (e.g., minimum class size or desired number)
min_count = class_counts.select("count").min()[0, 0]
print(f"\nMinimum records per class for safe sampling: {min_count}")

# Stratified sampling with safe check to avoid errors
df_balanced = pl.concat(
    [
        df_filtered.filter(pl.col("activity") == activity)
        .sample(n=min(min_count, df_filtered.filter(pl.col("activity") == activity).height), seed=42)
        for activity in df_filtered.select("activity").unique().to_series().to_list()
    ]
)

print("\nStratified Sampled DataFrame with Class Balance:")
print(df_balanced)


# In[72]:


#further reducing the dataset size
# Sample 50% of the dataset
df_balanced_sampled = df_balanced.sample(fraction=0.20, seed=42)

# Check the size of the new DataFrame
print(f"Original Size: {df_balanced.shape}")
print(f"Sampled Size: {df_balanced_sampled.shape}")


# In[73]:


# Count occurrences of each activity
class_counts = df_balanced_sampled.group_by("activity").agg(
    pl.col("activity").count().alias("count")
)

# Convert to Pandas for plotting
class_counts_pd = class_counts.to_pandas()

# Extract labels and counts
labels = class_counts_pd["activity"]
counts = class_counts_pd["count"]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(labels, counts, color="skyblue", edgecolor="black")

# Rotate and align x-axis labels
plt.xlabel("Activity")
plt.ylabel("Count")
plt.title("Activity Count Distribution")

# Rotate labels and align them to the right
plt.xticks(rotation=45, ha="right")

# Automatically adjust subplot to fit the labels
plt.tight_layout()

plt.show()


# In[74]:


# Drop specific columns
columns_to_drop = ["activity", "source_file"]
df_trimmed = df_balanced_sampled.drop(columns_to_drop)
print(df_trimmed)


# In[75]:


# Check column names
print(df_trimmed.columns)


# In[77]:


# List of columns to scale
columns_to_scale = ['time_acc', 'acc_x', 'acc_y', 'acc_z', 'time_gyro', 'gyro_x', 'gyro_y', 'gyro_z']


# Replace original columns with scaled values
# Apply Min-Max Scaling
df_scaled_replaced = df_balanced_sampled.with_columns(
    [((pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min())).alias(col)
     for col in columns_to_scale]
)

print("\nDataFrame with Replaced Scaled Columns:")
print(df_scaled_replaced)


# In[78]:


# Convert Polars DataFrame to Pandas for easier correlation
df_pandas = df_trimmed.to_pandas()

# Compute correlation matrix
correlation_matrix = df_pandas.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# In[79]:


# Plot correlation matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[80]:


#time_gyro & time_acc are having very high corelation as can be seen from heatmap , hence we can drop any one of them for the dataset

# Compute correlation matrix
correlation_matrix = df_pandas.corr().abs()

# Define correlation threshold
correlation_threshold = 0.9

# Identify columns to drop based on high correlation
to_drop = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > correlation_threshold:
            colname = correlation_matrix.columns[j]
            to_drop.add(colname)

print(f"\nColumns to drop due to high correlation: {to_drop}")

# Drop high-correlation columns in Polars
df_scaled_cleaned = df_scaled_replaced.drop(list(to_drop))

print("\nDataFrame after dropping highly correlated columns:")
print(df_scaled_cleaned)


# In[81]:


df_scaled_cleaned=df_scaled_cleaned.drop('source_file')


# In[82]:


df_scaled_cleaned.head()


# In[83]:


# Split features and target
X = df_scaled_cleaned.select(pl.exclude("activity")).to_numpy()  # Select all columns except "label"
y = df_scaled_cleaned["activity"].to_numpy()  # Target/label column



# In[86]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[87]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,stratify=y
)

print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")


# In[88]:


from sklearn.linear_model import LogisticRegression
# Create and train the Logistic Regression model
logisticregressionmodel = LogisticRegression(max_iter=1000)
logisticregressionmodel.fit(X_train, y_train)

print("Logistic Regression Model Trained Successfully!")


# In[89]:


# Make predictions
y_pred = logisticregressionmodel.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
print("\n Logistic Regression Model Classification Report:")
print(classification_report(y_test, y_pred))


# In[90]:


# Create and train the Random Forest model
randomforestmodel = RandomForestClassifier(n_estimators=100, random_state=42)
randomforestmodel.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")


# In[91]:


# Make predictions
y_pred = randomforestmodel.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f" Random Forest Model Accuracy: {accuracy:.2f}")
print("\n Random Forest Model Classification Report:")
print(classification_report(y_test, y_pred))


# In[92]:


from sklearn.svm import SVC
# Create and train the SVC model

svc_model = SVC(kernel='linear', C=1.0, random_state=42)
svc_model.fit(X_train, y_train)

print("SVC Model Trained Successfully!")


# In[93]:


# Make predictions
y_pred = svc_model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[95]:


#!pip install xgboost


# In[97]:


from sklearn.preprocessing import LabelEncoder
# Encode target labels (Convert categorical to numeric)
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[99]:


# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[101]:


import xgboost as xgb

# Create DMatrix for XGBoost (efficient training format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': len(set(y)),  # Number of classes
    'eval_metric': 'mlogloss',  # Log loss for multiclass classification
    'max_depth': 6,  # Maximum depth of a tree
    'eta': 0.1,  # Learning rate
    'gamma': 0.1,  # Minimum loss reduction required for split
}

# Train the XGBoost model
num_rounds = 100  # Number of boosting rounds
xgbmodel = xgb.train(params, dtrain, num_boost_round=num_rounds)
print("XGBoost Model Trained Successfully!")



# In[102]:


# Make predictions
y_pred = xgbmodel.predict(dtest)

# Evaluate model 
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:


#!pip install tensorflow


# In[103]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Define model architecture
dnnmodel = Sequential()
dnnmodel.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.3))  # Reduce overfitting

dnnmodel.add(Dense(64, activation='relu'))
dnnmodel.add(BatchNormalization())
dnnmodel.add(Dropout(0.3))

dnnmodel.add(Dense(32, activation='relu'))
dnnmodel.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer for multi-class classification

# Compile the model
dnnmodel.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
dnnmodel.summary()


# In[107]:


# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = dnnmodel.fit(
    X_train, y_train,
    epochs=25,  # Can be increased for better results
    batch_size=64,  # Larger batch sizes can improve performance
    callbacks=[early_stopping]
)


# In[108]:


# Train the model
#history = dnnmodel.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)


# In[109]:


# Evaluate the model on the test set
loss, accuracy = dnnmodel.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


# In[114]:


# Evaluate on test data
loss, accuracy = dnnmodel.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


# In[116]:


# Get predicted class labels
y_pred = np.argmax(dnnmodel.predict(X_test), axis=1)
# Generate classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)


# In[ ]:





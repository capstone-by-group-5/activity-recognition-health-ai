#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import zipfile
import os

# Path to your zip file
zip_file_path = 'Raw_time_domian_data.zip'

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('extracted_dataset')  # Extracts to 'extracted_dataset' folder

# Path to the extracted directory
extracted_path = 'extracted_dataset'


# In[3]:


# Column names to assign
column_names = [
    "time_acc", "acc_x", "acc_y", "acc_z",
    "time_gyro", "gyro_x", "gyro_y", "gyro_z"
]


# In[5]:


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


# In[6]:


combined_df.shape


# In[7]:


# Remove duplicates based on all columns
combined_df = combined_df.unique()


# In[8]:


combined_df.shape


# In[9]:


# Check for rows with any null values
rows_with_nulls = combined_df.filter(combined_df.select(pl.all().is_null()).sum_horizontal() > 0)
print(rows_with_nulls)


# In[10]:


# Drop rows with any null values
clean_df = combined_df.drop_nulls()
clean_df.shape


# In[11]:


print(clean_df['activity'].unique())


# In[12]:


# Remove numbers followed by a dot
clean_df = clean_df.with_columns(
    pl.col("activity").str.replace(r"\d+\.\s*", "").alias("activity")
)


# In[13]:


clean_df.shape


# In[14]:


# Count class occurrences
class_counts = clean_df.group_by("activity").agg(
    pl.col("activity").count().alias("count")
)

print(class_counts)


# In[15]:


# Drop rows where 'activity' equals 'walking'
df_filtered = clean_df.filter(clean_df["activity"] != "Table-tennis")


# In[16]:


df_filtered.shape


# In[17]:


# Convert to Pandas for plotting
class_counts_pd = class_counts.to_pandas()

# Extract labels and values for plotting
labels = class_counts_pd["activity"]
counts = class_counts_pd["count"]


# In[18]:


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


# In[19]:


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


# In[20]:


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


# In[38]:


#further reducing the dataset size
# Sample 50% of the dataset
df_balanced_sampled = df_filtered.sample(fraction=0.75, seed=42)

# Check the size of the new DataFrame
print(f"Original Size: {df_balanced.shape}")
print(f"Sampled Size: {df_balanced_sampled.shape}")


# In[21]:


#further reducing the dataset size
# Sample 50% of the dataset
df_balanced_sampled = df_balanced.sample(fraction=0.75, seed=42)

# Check the size of the new DataFrame
print(f"Original Size: {df_balanced.shape}")
print(f"Sampled Size: {df_balanced_sampled.shape}")


# In[39]:


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


# In[40]:


# Drop specific columns
columns_to_drop = ["activity", "source_file"]
df_trimmed = df_balanced_sampled.drop(columns_to_drop)
print(df_trimmed)


# In[41]:


# Check column names
print(df_trimmed.columns)


# In[42]:


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


# In[43]:


# Convert Polars DataFrame to Pandas for easier correlation
df_pandas = df_trimmed.to_pandas()

# Compute correlation matrix
correlation_matrix = df_pandas.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)


# In[44]:


# Plot correlation matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


# In[45]:


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


# In[46]:


df_scaled_cleaned=df_scaled_cleaned.drop('source_file')


# In[47]:


df_scaled_cleaned.head()


# In[48]:


# Split features and target
X = df_scaled_cleaned.select(pl.exclude("activity")).to_numpy()  # Select all columns except "label"
y = df_scaled_cleaned["activity"].to_numpy()  # Target/label column



# In[49]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[50]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,stratify=y
)

print(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, Shape of y_test: {y_test.shape}")


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


# In[95]:


#!pip install xgboost


# In[51]:


from sklearn.preprocessing import LabelEncoder
# Encode target labels (Convert categorical to numeric)
encoder = LabelEncoder()
y = encoder.fit_transform(y)


# In[52]:


# Split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[55]:


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
    'eta': 0.05,  # Learning rate
    'gamma': 0.1,  # Minimum loss reduction required for split
}

# Train the XGBoost model
num_rounds = 500  # Number of boosting rounds
xgbmodel = xgb.train(params, dtrain, num_boost_round=num_rounds)
print("XGBoost Model Trained Successfully!")



# In[56]:


# Make predictions
y_pred = xgbmodel.predict(dtest)

# Evaluate model 
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[ ]:





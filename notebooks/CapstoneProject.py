#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# pip install kaggle 
# pip install kagglehub
#run this above command if the kaggle is not installed on your machine


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")

print("Path to dataset files:", path)


# In[4]:


df=pd.read_csv(path+"/train.csv")
df_test=pd.read_csv(path+"/test.csv")


# In[5]:


df.head()


# In[6]:


df_test.head()


# In[7]:


df.shape


# In[8]:


df_test.shape


# In[9]:


df.info()


# In[10]:


missing_cols=df.columns[df.isnull().sum()>0]
# Print columns with missing values
print("Columns with Missing Values:\n", missing_cols)


# In[11]:


missing_cols=df.columns[df.isna().sum()>0]
# Print columns with missing values
print("Columns with Missing Values:\n", missing_cols)


# In[12]:


df.describe()


# In[13]:


len(df[df.duplicated()])


# In[14]:


print(df['Activity'].unique())


# In[15]:


#Check the Target Class Imbalance

# Count occurrences of each unique label
label_counts = df['Activity'].value_counts()

# Plot the pie chart
plt.figure(figsize=(6, 6))
plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140, colors=['lightblue', 'lightgreen', 'lightcoral'])
plt.title("Distribution of Activity")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.show()


# In[16]:


# Count the number of occurrences for each activity
activity_counts = df['Activity'].value_counts()

# Plot the bar chart
plt.figure(figsize=(10, 6))
activity_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Add labels and title
plt.title('Count of Samples per Activity in HAR Dataset')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# In[17]:


X_initial = df.iloc[:,:-1]
y_initial=df.iloc[:,-1]


# In[21]:


# Compute correlation matrix of the columns to remove the redundant columns present
corr_matrix = X_initial.corr()


# its difficult to visualize correlated max having 563 columns

# Compute the correlation matrix
corr_matrix_abs = corr_matrix.abs()  # Get absolute correlation values

# Create an upper triangular mask to avoid duplicate comparisons
upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

# Find columns with correlation > 0.97
to_drop = [column for column in upper.columns if any(upper[column] > 0.97)]

# Drop highly correlated columns
df_cleaned = df.drop(columns=to_drop)

print("Original DataFrame:\n", df.shape)
print("\nCorrelation Matrix:\n", corr_matrix_abs.shape)
print("\nColumns dropped:", to_drop)
print("\nCleaned DataFrame:\n", df_cleaned.shape)


# In[23]:


#dropping column found to be highly corelated to avoid redudancy
df_test_cleaned=df_test.drop(columns=['tBodyAcc-mad()-X', 'tBodyAcc-mad()-Y', 'tBodyAcc-mad()-Z', 'tBodyAcc-max()-X', 'tBodyAcc-sma()', 'tBodyAcc-iqr()-X', 'tBodyAcc-iqr()-Y', 'tBodyAcc-iqr()-Z', 'tGravityAcc-mad()-X', 'tGravityAcc-mad()-Y', 'tGravityAcc-mad()-Z', 'tGravityAcc-max()-X', 'tGravityAcc-max()-Y', 'tGravityAcc-max()-Z', 'tGravityAcc-min()-X', 'tGravityAcc-min()-Y', 'tGravityAcc-min()-Z', 'tGravityAcc-energy()-X', 'tGravityAcc-iqr()-X', 'tGravityAcc-iqr()-Y', 'tGravityAcc-iqr()-Z', 'tGravityAcc-arCoeff()-X,2', 'tGravityAcc-arCoeff()-X,3', 'tGravityAcc-arCoeff()-X,4', 'tGravityAcc-arCoeff()-Y,2', 'tGravityAcc-arCoeff()-Y,3', 'tGravityAcc-arCoeff()-Y,4', 'tGravityAcc-arCoeff()-Z,2', 'tGravityAcc-arCoeff()-Z,3', 'tGravityAcc-arCoeff()-Z,4', 'tBodyAccJerk-std()-X', 'tBodyAccJerk-mad()-X', 'tBodyAccJerk-mad()-Y', 'tBodyAccJerk-mad()-Z', 'tBodyAccJerk-sma()', 'tBodyAccJerk-iqr()-X', 'tBodyAccJerk-iqr()-Y', 'tBodyAccJerk-iqr()-Z', 'tBodyAccJerk-entropy()-Y', 'tBodyAccJerk-entropy()-Z', 'tBodyGyro-mad()-X', 'tBodyGyro-mad()-Y', 'tBodyGyro-mad()-Z', 'tBodyGyro-iqr()-X', 'tBodyGyro-iqr()-Y', 'tBodyGyro-iqr()-Z', 'tBodyGyroJerk-mad()-X', 'tBodyGyroJerk-mad()-Y', 'tBodyGyroJerk-mad()-Z', 'tBodyGyroJerk-max()-X', 'tBodyGyroJerk-max()-Z', 'tBodyGyroJerk-sma()', 'tBodyGyroJerk-iqr()-X', 'tBodyGyroJerk-iqr()-Y', 'tBodyGyroJerk-iqr()-Z', 'tBodyGyroJerk-entropy()-Z', 'tBodyGyroJerk-arCoeff()-Z,1', 'tBodyAccMag-mean()', 'tBodyAccMag-mad()', 'tBodyAccMag-max()', 'tBodyAccMag-sma()', 'tBodyAccMag-iqr()', 'tGravityAccMag-mean()', 'tGravityAccMag-std()', 'tGravityAccMag-mad()', 'tGravityAccMag-max()', 'tGravityAccMag-min()', 'tGravityAccMag-sma()', 'tGravityAccMag-energy()', 'tGravityAccMag-iqr()', 'tGravityAccMag-entropy()', 'tGravityAccMag-arCoeff()1', 'tGravityAccMag-arCoeff()2', 'tGravityAccMag-arCoeff()3', 'tGravityAccMag-arCoeff()4', 'tBodyAccJerkMag-mean()', 'tBodyAccJerkMag-std()', 'tBodyAccJerkMag-mad()', 'tBodyAccJerkMag-max()', 'tBodyAccJerkMag-sma()', 'tBodyAccJerkMag-energy()', 'tBodyAccJerkMag-iqr()', 'tBodyAccJerkMag-entropy()', 'tBodyGyroMag-mean()', 'tBodyGyroMag-mad()', 'tBodyGyroMag-max()', 'tBodyGyroMag-sma()', 'tBodyGyroMag-iqr()', 'tBodyGyroJerkMag-mean()', 'tBodyGyroJerkMag-std()', 'tBodyGyroJerkMag-mad()', 'tBodyGyroJerkMag-max()', 'tBodyGyroJerkMag-sma()', 'tBodyGyroJerkMag-iqr()', 'tBodyGyroJerkMag-entropy()', 'fBodyAcc-mean()-X', 'fBodyAcc-mean()-Y', 'fBodyAcc-mean()-Z', 'fBodyAcc-std()-X', 'fBodyAcc-std()-Y', 'fBodyAcc-std()-Z', 'fBodyAcc-mad()-X', 'fBodyAcc-mad()-Y', 'fBodyAcc-mad()-Z', 'fBodyAcc-max()-X', 'fBodyAcc-max()-Y', 'fBodyAcc-max()-Z', 'fBodyAcc-sma()', 'fBodyAcc-energy()-X', 'fBodyAcc-energy()-Z', 'fBodyAcc-iqr()-Z', 'fBodyAcc-entropy()-X', 'fBodyAcc-entropy()-Y', 'fBodyAcc-entropy()-Z', 'fBodyAcc-kurtosis()-X', 'fBodyAcc-kurtosis()-Y', 'fBodyAcc-kurtosis()-Z', 'fBodyAcc-bandsEnergy()-1,8', 'fBodyAcc-bandsEnergy()-1,16', 'fBodyAcc-bandsEnergy()-17,32', 'fBodyAcc-bandsEnergy()-33,48', 'fBodyAcc-bandsEnergy()-49,64', 'fBodyAcc-bandsEnergy()-1,24', 'fBodyAcc-bandsEnergy()-25,48', 'fBodyAcc-bandsEnergy()-1,16.1', 'fBodyAcc-bandsEnergy()-17,32.1', 'fBodyAcc-bandsEnergy()-33,48.1', 'fBodyAcc-bandsEnergy()-49,64.1', 'fBodyAcc-bandsEnergy()-1,24.1', 'fBodyAcc-bandsEnergy()-25,48.1', 'fBodyAcc-bandsEnergy()-1,16.2', 'fBodyAcc-bandsEnergy()-17,32.2', 'fBodyAcc-bandsEnergy()-33,48.2', 'fBodyAcc-bandsEnergy()-49,64.2', 'fBodyAcc-bandsEnergy()-1,24.2', 'fBodyAcc-bandsEnergy()-25,48.2', 'fBodyAccJerk-mean()-X', 'fBodyAccJerk-mean()-Y', 'fBodyAccJerk-mean()-Z', 'fBodyAccJerk-std()-X', 'fBodyAccJerk-std()-Y', 'fBodyAccJerk-std()-Z', 'fBodyAccJerk-mad()-X', 'fBodyAccJerk-mad()-Y', 'fBodyAccJerk-mad()-Z', 'fBodyAccJerk-max()-X', 'fBodyAccJerk-max()-Y', 'fBodyAccJerk-max()-Z', 'fBodyAccJerk-sma()', 'fBodyAccJerk-energy()-X', 'fBodyAccJerk-energy()-Y', 'fBodyAccJerk-energy()-Z', 'fBodyAccJerk-iqr()-X', 'fBodyAccJerk-iqr()-Y', 'fBodyAccJerk-iqr()-Z', 'fBodyAccJerk-entropy()-X', 'fBodyAccJerk-entropy()-Y', 'fBodyAccJerk-entropy()-Z', 'fBodyAccJerk-bandsEnergy()-1,8', 'fBodyAccJerk-bandsEnergy()-9,16', 'fBodyAccJerk-bandsEnergy()-17,24', 'fBodyAccJerk-bandsEnergy()-25,32', 'fBodyAccJerk-bandsEnergy()-17,32', 'fBodyAccJerk-bandsEnergy()-49,64', 'fBodyAccJerk-bandsEnergy()-1,24', 'fBodyAccJerk-bandsEnergy()-9,16.1', 'fBodyAccJerk-bandsEnergy()-17,24.1', 'fBodyAccJerk-bandsEnergy()-25,32.1', 'fBodyAccJerk-bandsEnergy()-1,16.1', 'fBodyAccJerk-bandsEnergy()-17,32.1', 'fBodyAccJerk-bandsEnergy()-49,64.1', 'fBodyAccJerk-bandsEnergy()-1,24.1', 'fBodyAccJerk-bandsEnergy()-25,48.1', 'fBodyAccJerk-bandsEnergy()-9,16.2', 'fBodyAccJerk-bandsEnergy()-17,24.2', 'fBodyAccJerk-bandsEnergy()-25,32.2', 'fBodyAccJerk-bandsEnergy()-33,40.2', 'fBodyAccJerk-bandsEnergy()-1,16.2', 'fBodyAccJerk-bandsEnergy()-17,32.2', 'fBodyAccJerk-bandsEnergy()-33,48.2', 'fBodyAccJerk-bandsEnergy()-49,64.2', 'fBodyAccJerk-bandsEnergy()-1,24.2', 'fBodyAccJerk-bandsEnergy()-25,48.2', 'fBodyGyro-mean()-X', 'fBodyGyro-mean()-Y', 'fBodyGyro-mean()-Z', 'fBodyGyro-std()-X', 'fBodyGyro-std()-Y', 'fBodyGyro-std()-Z', 'fBodyGyro-mad()-X', 'fBodyGyro-mad()-Y', 'fBodyGyro-mad()-Z', 'fBodyGyro-max()-X', 'fBodyGyro-max()-Z', 'fBodyGyro-sma()', 'fBodyGyro-energy()-Y', 'fBodyGyro-energy()-Z', 'fBodyGyro-entropy()-X', 'fBodyGyro-entropy()-Y', 'fBodyGyro-entropy()-Z', 'fBodyGyro-kurtosis()-X', 'fBodyGyro-kurtosis()-Y', 'fBodyGyro-kurtosis()-Z', 'fBodyGyro-bandsEnergy()-1,8', 'fBodyGyro-bandsEnergy()-1,16', 'fBodyGyro-bandsEnergy()-17,32', 'fBodyGyro-bandsEnergy()-33,48', 'fBodyGyro-bandsEnergy()-49,64', 'fBodyGyro-bandsEnergy()-1,24', 'fBodyGyro-bandsEnergy()-25,48', 'fBodyGyro-bandsEnergy()-17,32.1', 'fBodyGyro-bandsEnergy()-33,48.1', 'fBodyGyro-bandsEnergy()-49,64.1', 'fBodyGyro-bandsEnergy()-1,24.1', 'fBodyGyro-bandsEnergy()-25,48.1', 'fBodyGyro-bandsEnergy()-1,16.2', 'fBodyGyro-bandsEnergy()-17,32.2', 'fBodyGyro-bandsEnergy()-33,48.2', 'fBodyGyro-bandsEnergy()-49,64.2', 'fBodyGyro-bandsEnergy()-1,24.2', 'fBodyGyro-bandsEnergy()-25,48.2', 'fBodyAccMag-mean()', 'fBodyAccMag-std()', 'fBodyAccMag-mad()', 'fBodyAccMag-max()', 'fBodyAccMag-sma()', 'fBodyAccMag-iqr()', 'fBodyAccMag-entropy()', 'fBodyAccMag-kurtosis()', 'fBodyBodyAccJerkMag-mean()', 'fBodyBodyAccJerkMag-std()', 'fBodyBodyAccJerkMag-mad()', 'fBodyBodyAccJerkMag-max()', 'fBodyBodyAccJerkMag-sma()', 'fBodyBodyAccJerkMag-energy()', 'fBodyBodyAccJerkMag-iqr()', 'fBodyBodyAccJerkMag-entropy()', 'fBodyBodyAccJerkMag-kurtosis()', 'fBodyBodyGyroMag-mean()', 'fBodyBodyGyroMag-std()', 'fBodyBodyGyroMag-mad()', 'fBodyBodyGyroMag-max()', 'fBodyBodyGyroMag-sma()', 'fBodyBodyGyroMag-entropy()', 'fBodyBodyGyroMag-kurtosis()', 'fBodyBodyGyroJerkMag-mean()', 'fBodyBodyGyroJerkMag-std()', 'fBodyBodyGyroJerkMag-mad()', 'fBodyBodyGyroJerkMag-max()', 'fBodyBodyGyroJerkMag-sma()', 'fBodyBodyGyroJerkMag-energy()', 'fBodyBodyGyroJerkMag-iqr()', 'fBodyBodyGyroJerkMag-entropy()', 'angle(X,gravityMean)', 'angle(Y,gravityMean)', 'angle(Z,gravityMean)'])


# In[24]:


df_test_cleaned.shape


# In[25]:


df_test_cleaned.describe()


# In[26]:


X_train = df_cleaned.iloc[:,:-1]
y_train =df_cleaned.iloc[:,-1]

X_test=df_test_cleaned.iloc[:,:-1]
y_test=df_test_cleaned.iloc[:,-1]


# In[27]:


from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()

# Fit and transform on X_train
#NOT USING SCALING on TRAIN DATA AS DATA IS ALREADY SCALED
X_train_scaled = scaler.fit_transform(X_train)

# Transform X_test using the same scaler
X_test_scaled = scaler.transform(X_test)



# In[28]:


from sklearn.decomposition import PCA
pca = PCA(n_components=0.95) 
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)

# Check the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Number of components to retain 95% variance: {pca.n_components_}")



# In[29]:


X_train_pca.shape


# In[30]:


X_test_pca.shape


# In[31]:


#Outliers removal based on z score

from scipy import stats
# Calculate Z-scores for each feature
z_scores = np.abs(stats.zscore(X_train_pca))

# Set a threshold (e.g., 3 standard deviations)
threshold = 3
outliers_z = np.where(z_scores > threshold)

# Count and print the number of outliers
print(f"Number of outliers detected by Z-Score: {len(np.unique(outliers_z[0]))}")

# Optional: Remove outliers
X_no_outliers_z = X_train_pca[(z_scores < threshold).all(axis=1)]
y_no_outliers_z = y_train[(z_scores < threshold).all(axis=1)]


# In[32]:


X_no_outliers_z.shape


# In[33]:


y_no_outliers_z.shape


# In[34]:


# assigning proper columns to train dataset 
#X_train = X_no_outliers_z
#y_train= y_no_outliers_z

X_train = X_no_outliers_z
y_train= y_no_outliers_z

X_test=X_test_pca
y_test=y_test


# In[35]:


X_train.shape


# In[36]:


y_train.shape


# In[37]:


X_test.shape


# In[38]:


from sklearn import svm
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[39]:


from sklearn.linear_model import LogisticRegression
# Create and train the Logistic Regression model
logisticregressionmodel = LogisticRegression(max_iter=1000)
logisticregressionmodel.fit(X_train, y_train)

print("Logistic Regression Model Trained Successfully!")


# In[40]:


# Make predictions
y_pred = logisticregressionmodel.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {accuracy:.2f}")
print("\n Logistic Regression Model Classification Report:")
print(classification_report(y_test, y_pred))


# In[41]:


svc_linear = svm.SVC(kernel='linear')
svc_linear.fit(X_train, y_train)
# make predictions on test set
y_pred=svc_linear.predict(X_test)
# compute and print accuracy score
print('SVC Linear: Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print('\n\nConfusion matrix\n\n', cm)


# In[42]:


svc_poly = svm.SVC(kernel='poly')
svc_poly.fit(X_train, y_train)
# make predictions on test set
y_pred=svc_poly.predict(X_test)
# compute and print accuracy score
print('SVC poly: Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
print("\nClassification Report:\n\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print('\n\nConfusion matrix\n\n', cm)


# In[43]:


svc_rbf = svm.SVC(kernel='rbf')
svc_rbf.fit(X_train, y_train)
# make predictions on test set
y_pred=svc_rbf.predict(X_test)
# compute and print accuracy score
print('SVC RBF: Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

print("\nClassification Report:\n\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print('\n\nConfusion matrix\n\n', cm)


# In[44]:


from sklearn.preprocessing import LabelEncoder
# Encode target labels (Convert categorical to numeric)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# In[45]:


import xgboost as xgb

# Create DMatrix for XGBoost (efficient training format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set XGBoost parameters
params = {
    'objective': 'multi:softmax',  # Multiclass classification
    'num_class': len(set(y_train)),  # Number of classes
    'eval_metric': 'mlogloss',  # Log loss for multiclass classification
    'max_depth': 6,  # Maximum depth of a tree
    'eta': 0.1,  # Learning rate
    'gamma': 0.1,  # Minimum loss reduction required for split
}

# Train the XGBoost model
num_rounds = 100  # Number of boosting rounds
xgbmodel = xgb.train(params, dtrain, num_boost_round=num_rounds)
print("XGBoost Model Trained Successfully!")


# In[46]:


# Make predictions
y_pred = xgbmodel.predict(dtest)

# Evaluate model 
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[47]:


from sklearn.ensemble import RandomForestClassifier
# Create and train the Random Forest model
randomforestmodel = RandomForestClassifier(n_estimators=100, random_state=42)
randomforestmodel.fit(X_train, y_train)

print("Random Forest Model Trained Successfully!")


# In[48]:


# Make predictions
y_pred = randomforestmodel.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f" Random Forest Model Accuracy: {accuracy:.2f}")
print("\n Random Forest Model Classification Report:")
print(classification_report(y_test, y_pred))


# In[49]:


import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[51]:


#Now lets apply the deep learning Model 

# Define model architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_train)), activation='softmax')  # Softmax for multi-class classification
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()


# In[52]:


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)


# In[53]:


# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# In[54]:


# Predict class labels
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted Classes:", encoder.inverse_transform(predicted_classes[:10]))
print("Actual Classes:", encoder.inverse_transform(y_test[:10]))


# In[ ]:





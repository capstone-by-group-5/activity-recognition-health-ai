#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense,Dropout
import os
import warnings
from tensorflow import keras
from tensorflow.keras import layers


train_data = pd.read_csv('/Users/apple/Documents/projects/activity-recognition-health-ai/data/raw/train.csv')
test_data = pd.read_csv('/Users/apple/Documents/projects/activity-recognition-health-ai/data/raw/test.csv')

print(f'Shape of train data is: {train_data.shape}\nShape of test data is: {test_data.shape}')


# In[ ]:





# In[3]:


pd.set_option("display.max_columns", None)


# In[4]:


# train_data.head()


# In[5]:


train_data.columns


# In[6]:


# train_data.describe()


# In[7]:


train_data['Activity'].unique()


# In[8]:


train_data['Activity'].value_counts().sort_values().plot(kind = 'bar', color = 'pink')


# Handling Missing and Duplicate Data:

# In[9]:


# Check for missing values and duplicates:


print("Missing values in train data:", train_data.isnull().sum().sum())
print("Missing values in test data:", test_data.isnull().sum().sum())
print("Duplicate rows in train data:", train_data.duplicated().sum())
print("Duplicate rows in test data:", test_data.duplicated().sum())


# In[10]:


train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()


# In[11]:


# Separate features and target

X_train = train_data.iloc[:, :-2]  # Exclude 'subject' and 'Activity'
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-2]  # Exclude 'subject' and 'Activity'
y_test = test_data.iloc[:, -1]


# In[ ]:





# In[ ]:





# In[12]:


def remove_outliers_modified_zscore(df, y, threshold=3.5):
    median = df.median()
    mad = (df - median).abs().median()
    modified_z_score = 0.6745 * (df - median) / mad
    mask = (modified_z_score.abs() < threshold).all(axis=1)
    return df[mask], y[mask]

# X_train, y_train = remove_outliers_modified_zscore(X_train, y_train)



def remove_outliers_iqr(df, y, columns, max_outlier_fraction=0.1):
    """Removes rows with excessive outliers based on the IQR method."""
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))
    
    # Count number of outlier features per row
    outlier_counts = outlier_mask.sum(axis=1)
    
    # Keep rows where fewer than 10% of features are outliers
    mask = outlier_counts <= (max_outlier_fraction * len(columns))
    
    return df[mask], y[mask]

# X_train, y_train = remove_outliers_iqr(X_train, y_train, X_train.columns)



# In[13]:

X_train, y_train = remove_outliers_iqr(X_train, y_train, X_train.columns, max_outlier_fraction=0.1)


# In[ ]:





# In[14]:


# Encode activity labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)


# Ensure y_test only contains labels present in y_train
y_test_filtered = y_test[y_test.isin(le.classes_)]

# Now transform y_test safely
y_test_encoded = le.transform(y_test_filtered)


# In[15]:


# Feature Selection:
# Select top 100 features using mutual information:


from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=100)
selector.fit(X_train, y_train_encoded)
x_train_selected = selector.transform(X_train)
x_test_selected = selector.transform(X_test)

# This reduces dimensionality, potentially improving model performance and training speed.


# In[ ]:





# In[16]:


# Normalize features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(x_train_selected)
X_test_scaled = scaler.transform(x_test_selected)


# In[ ]:





# In[17]:


# Print Shapes
print("Train Shape:", X_train_scaled.shape, y_train_encoded.shape)
print("Test Shape:", X_test_scaled.shape, y_test_encoded.shape)


# In[18]:


print(f"Original Training Samples: {train_data.shape[0]}")
print(f"Samples After Outlier Removal: {X_train.shape[0]}")
print(f"Samples Removed: {train_data.shape[0] - X_train.shape[0]}")


# In[19]:


print("Activity Labels in Train Set After Outlier Removal:")
print(y_train.value_counts())

print("\nActivity Labels in Test Set:")
print(y_test.value_counts())

print("\nActivity Labels Remaining After Filtering Test Set:")
print(y_test_filtered.value_counts())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





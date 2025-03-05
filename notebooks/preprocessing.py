#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os

data_dir = "data/raw"
train_path = os.path.join(data_dir, "train.csv")

def UploadFile(file_path):
    try:
        ds = pd.read_csv(file_path)
        print("File uploaded successfully ", file_path)
        return ds
    except:
        print("Please check CSV file ", file_path)
        return None
    

# Function to find missing values
def FindMissingValue(ds):
    missing = ds.isnull().sum()
    missing = missing[missing > 0]
    return missing

# Function to find duplicate rows
def FindDuplicates(ds):
    duplicates = ds.duplicated().sum()
    return duplicates

# Function to check for invalid data types
def FindDataType(ds):
    types = ds.dtypes
    return types
        
# Load datasets
trainDataSet = UploadFile(train_path)
        
if trainDataSet is not None:
        print(trainDataSet.head())
        print(trainDataSet.info())
        print(trainDataSet.describe())
        print(trainDataSet.nunique())
        
print("Unique Activity")
print("-------")
print(trainDataSet['Activity'].unique())
print("-------")

# Checking missing values
missing_values = FindMissingValue(trainDataSet)
print("-------")
print("Missing Values:")
print(missing_values)
print("-------")


# Checking duplicate rows
duplicate_rows = FindDuplicates(trainDataSet)
print("-------")
print("\nDuplicate Rows:")
print("-------")
print(duplicate_rows)



# Checking data types
data_types = FindDataType(trainDataSet)
print("-------")
print("\nData Types:")
print(data_types)
print("-------")

# Count occurrences of each unique label
label_counts = trainDataSet['Activity'].value_counts()

print(label_counts)


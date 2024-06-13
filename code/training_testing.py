#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import python library
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import os


# In[12]:


# S1 setup 

dir_path = os.path.abspath('./preprocessing/preprocessed_amp') # raw data folder change the path as required 
dir_files= os.listdir(dir_path)

train=[]
train_label=[]
test=[]
test_label=[]
for i in dir_files: 
    # print(i)
    files= os.listdir(f'{dir_path}/{i}')
    files.sort()
    # print(files)
    for j in files:
        if 'amp' in j: 
            df = pd.read_csv(f'{dir_path}/{i}/{j}')
            # print(df.isnull().any(axis=1).sum())
            X = df.iloc[0:, 0:114]
            y= df['headlabel']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train.append(X_train)
            train_label.append(y_train)
            test.append(X_test)
            test_label.append(y_test)

X_train= pd.concat(train, axis=0)
X_train=X_train.reset_index(drop=True)
X_train.to_csv(os.path.abspath('./training_testing_data/S1/X_train.csv'), index=False)

X_test= pd.concat(test, axis=0)
X_test=X_test.reset_index(drop=True)
X_test.to_csv(os.path.abspath('./training_testing_data/S1/X_test.csv'), index=False)

y_train= pd.concat(train_label, axis=0)
y_train=y_train.reset_index(drop=True)
y_train.to_csv(os.path.abspath('./training_testing_data/S1/y_train.csv'), index=False)

y_test= pd.concat(test_label, axis=0)
y_test=y_test.reset_index(drop=True)
y_test.to_csv(os.path.abspath('./training_testing_data/S1/y_test.csv'), index=False)


# In[3]:


# S4 setup 

# training data 
dir_path = os.path.abspath('./preprocessing/preprocessed_amp') # raw data folder change the path as required 
dir_files= os.listdir(dir_path)

train=[]
train_label=[]
semi=1
wild=1
for i in dir_files[:semi]: # change here for making wild as training
    # print(i)
    files= os.listdir(f'{dir_path}/{i}')
    files.sort()
    # print(files)
    for j in files:
        if 'amp' in j: 
            df = pd.read_csv(f'{dir_path}/{i}/{j}')
            # print(df.isnull().any(axis=1).sum())
            X = df.iloc[0:, 0:114]
            y= df['headlabel']

            train.append(X)
            train_label.append(y)
test=[]
test_label=[]

for i in dir_files[wild:]: # change here for making semi as testing
    # print(i)
    files= os.listdir(f'{dir_path}/{i}')
    files.sort()
    # print(files)
    for j in files:
        if 'amp' in j: 
            df = pd.read_csv(f'{dir_path}/{i}/{j}')
            # print(df.isnull().any(axis=1).sum())
            X = df.iloc[0:, 0:114]
            y= df['headlabel']
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            test.append(X)
            test_label.append(y)


X_train= pd.concat(train, axis=0)
X_train=X_train.reset_index(drop=True)
X_train.to_csv(os.path.abspath('./training_testing_data/S4/X_train.csv'), index=False)

X_test= pd.concat(test, axis=0)
X_test=X_test.reset_index(drop=True)
X_test.to_csv(os.path.abspath('./training_testing_data/S4/X_test.csv'), index=False)

y_train= pd.concat(train_label, axis=0)
y_train=y_train.reset_index(drop=True)
y_train.to_csv(os.path.abspath('./training_testing_data/S4/y_train.csv'), index=False)

y_test= pd.concat(test_label, axis=0)
y_test=y_test.reset_index(drop=True)
y_test.to_csv(os.path.abspath('./training_testing_data/S4/y_test.csv'), index=False)


# In[ ]:


# S3 setup 
import random
import re
# training data 
dir_path = os.path.abspath('./preprocessing/preprocessed_amp')  
dir_files= os.listdir(dir_path)
file_list=[]
for i in dir_files:
    sub_dir= f'{dir_path}/{i}'
    sub_list= os.listdir(sub_dir)
    sub_list = [os.path.join(f'{sub_dir}', file) for file in os.listdir(sub_dir)]
    file_list.append(sub_list)
complete_list = file_list[0]+file_list[1]

# Function to filter file names based on participant ID range
def filter_files_by_participant(file_names, start_id, end_id):
    pattern = re.compile(r'P(\d+)_')
    filtered_files = [file for file in file_names if start_id <= int(pattern.search(file).group(1)) <= end_id]
    return filtered_files

p1_to_p22_files = filter_files_by_participant(complete_list, 1, 22)
p23_to_p33_files = filter_files_by_participant(complete_list, 23, 33)

def select_random_files(files, num_files):
    if num_files > len(files):
        raise ValueError("num_files exceeds the length of the list")
    return random.sample(files, num_files)
list_length1= len(p1_to_p22_files)
list_length2= len(p23_to_p33_files)
training_1 = select_random_files(p1_to_p22_files, int(abs(list_length1*0.80))) # change num_files as per requirement 
training_2 = select_random_files(p23_to_p33_files,  int(abs(list_length2*0.80)) ) # change num_files as per requirement
testing_1 = [file for file in complete_list if file not in training_1]
testing_2 = [file for file in complete_list if file not in training_2]
training = training_1+training_2
testing = testing_1+testing_2

train=[]
train_label=[]
semi=1
wild=1

for j in training:
    if 'amp' in j: 
        df = pd.read_csv(f'{j}')
        # print(df.isnull().any(axis=1).sum())
        X = df.iloc[0:, 0:114]
        y= df['headlabel']

        train.append(X)
        train_label.append(y)
test=[]
test_label=[]

for j in testing:
    if 'amp' in j: 
        df = pd.read_csv(f'{j}')
        # print(df.isnull().any(axis=1).sum())
        X = df.iloc[0:, 0:114]
        y= df['headlabel']
        test.append(X)
        test_label.append(y)


X_train= pd.concat(train, axis=0)
X_train=X_train.reset_index(drop=True)
X_train.to_csv(os.path.abspath('./training_testing_data/S3/X_train.csv'), index=False)

X_test= pd.concat(test, axis=0)
X_test=X_test.reset_index(drop=True)
X_test.to_csv(os.path.abspath('./training_testing_data/S3/X_test.csv'), index=False)

y_train= pd.concat(train_label, axis=0)
y_train=y_train.reset_index(drop=True)
y_train.to_csv(os.path.abspath('./training_testing_data/S3/y_train.csv'), index=False)

y_test= pd.concat(test_label, axis=0)
y_test=y_test.reset_index(drop=True)
y_test.to_csv(os.path.abspath('./training_testing_data/S3/y_test.csv'), index=False)


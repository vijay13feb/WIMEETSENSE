#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os 
import joblib as jb
from collections import defaultdict
from itertools import groupby
from math import sqrt, atan2
import matplotlib.pyplot as plt
import pywt
from scipy.signal import savgol_filter
import pickle
import re


# Combine semi controlled CSI files

# In[ ]:


check_list = os.listdir(os.path.abspath('./preprocess/semi'))
check_list.sort()
first_names = [re.match(r'([^_]+)_', filename).group(1) for filename in check_list]

# Get unique first names
unique_first_names = set(first_names)
unique_first_names

dir_list = os.path.abspath('./preprocess/semi')
save= os.path.abspath('./input_combine/semi')
list_dir = os.listdir(dir_list)
list_dir.sort()
name=[] 
for k in unique_first_names: 
    os.makedirs(f'{save}/{k}', exist_ok=True)
    for i in list_dir:
        temp = i.split("_")
        name.append(temp[0])
    name_unique= np.unique(name)
    combine_list=[]
    for i in list_dir:
        if k in i:
            complete_path = f'{dir_list}/{i}'
            with open(complete_path, 'rb') as f:
                result = pickle.load(f)
            # column=result.shape[1]
            # print(column)
            len= result.shape[0]
            len=int(abs(0.03*len))
            # print(result.shape)
            combine_list.append(result)
        else:
            pass
    df = pd.concat(combine_list, axis=0)
    # df= df.drop(columns=['timestamp'], axis=1)
    df= df.reset_index(drop=True)
    print(df.shape)
    if 'headlabel' in df.columns:
        print('here')
        group = df.groupby('headlabel')
        label_name = list(group.groups.keys())
        for i in label_name:
            df_temp = group.get_group(i)
            df_temp=df_temp.reset_index(drop=True)
            df_temp = df_temp.iloc[0:,0:114]
            complete_path= f'{save}/{k}/{i}'
            with open(complete_path, 'wb') as f:
                pickle.dump(df_temp,f )
            


# Combine in-the-wild CSI files 

# In[ ]:


check_list = os.listdir(os.path.abspath('./preprocess/wild'))
check_list.sort()
first_names = [re.match(r'([^_]+)_', filename).group(1) for filename in check_list]

# Get unique first names
unique_first_names = set(first_names)
unique_first_names

dir_list = os.path.abspath('./preprocess/wild')
save= os.path.abspath('./input_combine/wild')
list_dir = os.listdir(dir_list)
list_dir.sort()
name=[] 
for k in unique_first_names: 
    os.makedirs(f'{save}/{k}', exist_ok=True)
    for i in list_dir:
        temp = i.split("_")
        name.append(temp[0])
    name_unique= np.unique(name)
    combine_list=[]
    for i in list_dir:
        if k in i:
            complete_path = f'{dir_list}/{i}'
            with open(complete_path, 'rb') as f:
                result = pickle.load(f)
            # column=result.shape[1]
            # print(column)
            len= result.shape[0]
            len=int(abs(0.03*len))
            # print(result.shape)
            combine_list.append(result)
        else:
            pass
    df = pd.concat(combine_list, axis=0)
    # df= df.drop(columns=['timestamp'], axis=1)
    df= df.reset_index(drop=True)
    print(df.shape)
    if 'headlabel' in df.columns:
        print('here')
        group = df.groupby('headlabel')
        label_name = list(group.groups.keys())
        for i in label_name:
            df_temp = group.get_group(i)
            df_temp=df_temp.reset_index(drop=True)
            df_temp = df_temp.iloc[0:,0:114]
            complete_path= f'{save}/{k}/{i}'
            with open(complete_path, 'wb') as f:
                pickle.dump(df_temp,f )


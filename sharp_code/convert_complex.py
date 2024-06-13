#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import numpy as np
import os
import sys
import pickle
import joblib as jb


# In[2]:


#/home/vijay/paper_jc/Neurips_raw_csi_data/code/raw_csi
save_path = os.path.abspath('./preprocess')
dir_path = os.path.abspath('./code/raw_csi')
directories = dir_path.split('/')
directories.remove("sharp_code")
dir_path  = '/'.join(directories)


# In[34]:


# directory 
# computing  
dir_files= os.listdir(dir_path)
amp_files=[]
label_files=[]
for i in dir_files: 

    files= os.listdir(f'{dir_path}/{i}')

    for j in files:
        
        result = pd.read_csv(f'{dir_path}/{i}/{j}')
        len_= result.shape[0]
        len_=int(abs(0.03*len_))
        result=result.iloc[len_:-len_, 128:] # remove first 128 rows.
        new_data = result.iloc[:, :-2] 
        new_data.columns= range(len(new_data.columns))
        delete_idxs =  np.asarray([0,1,2,3,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,254,255])
        new_data = new_data.drop(new_data.columns[delete_idxs], axis=1)
        new_data.columns= range(len(new_data.columns))
    
        ct=[]
        for K in range(1, new_data.shape[1], 2):
            complex_val = new_data[K] + 1j * new_data[K-1]
            ct.append(complex_val) 

        final_ans = pd.concat(ct,axis=1)

        final_ans["headlabel"] = result["headlabel"]
        complete_path = f'{save_path}/{i}/{j[:-4]}.txt'
        with open (complete_path, 'wb') as f:
            pickle.dump(final_ans, f)


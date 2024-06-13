#!/usr/bin/env python
# coding: utf-8

# In[12]:


import argparse
import numpy as np
import scipy.io as sio
from os import listdir
import joblib
import pandas as pd
import os
import pickle


# In[15]:





# In[36]:


# preprocessing the input combine files: 
dir_path = os.path.abspath('./input_combine/semi')
save_path = os.path.abspath('./preprocessing/semi')
dir_files= os.listdir(dir_path)
for i in dir_files:
    os.makedirs(f'{save_path}/{i}', exist_ok=True)
    files_path = f'{dir_path}/{i}'
    files = os.listdir(files_path)
    for j in files:
        path = f'{dir_path}/{i}/{j}'
        with open(path, 'rb') as f:
                csi_buff = pickle.load(f)
        
        csi_buff = csi_buff.iloc[:, :]
        length = csi_buff.shape[0]
        length2 = csi_buff.shape[1]
        csi_buff = np.fft.fftshift(csi_buff, axes=1)
        delete_idxs = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
        csi_buff = np.delete(csi_buff, delete_idxs, axis=0)
        n_ss = 1
        n_core = 1
        n_tot = n_ss * n_core
        start =0
        end = int(np.floor(csi_buff.shape[0]/n_tot))
        signal_complete = np.zeros((csi_buff.shape[1] - delete_idxs.shape[0], end-start, n_tot), dtype=complex)
        stream=0
        signal_stream = csi_buff[stream:end*n_tot + 1:n_tot, :][start:end, :]
        signal_stream = np.delete(signal_stream, delete_idxs, axis=1)
        mean_signal = np.mean(np.abs(signal_stream), axis=1, keepdims=True)
        H_m = signal_stream/mean_signal
        signal_complete[:, :, stream] = H_m.T
        with open (f"{save_path}/{i}/{j}.txt", 'wb') as f:
            pickle.dump(signal_complete, f)
      
    


# In[ ]:


# preprocessing the input combine files: 
dir_path = os.path.abspath('./input_combine/wild')
save_path = os.path.abspath('./preprocessing/wild')
dir_files= os.listdir(dir_path)
for i in dir_files:
    os.makedirs(f'{save_path}/{i}', exist_ok=True)
    files_path = f'{dir_path}/{i}'
    files = os.listdir(files_path)
    for j in files:
        path = f'{dir_path}/{i}/{j}'
        with open(path, 'rb') as f:
                csi_buff = pickle.load(f)
        
        csi_buff = csi_buff.iloc[:, :]
        length = csi_buff.shape[0]
        length2 = csi_buff.shape[1]
        csi_buff = np.fft.fftshift(csi_buff, axes=1)
        delete_idxs = np.argwhere(np.sum(csi_buff, axis=1) == 0)[:, 0]
        csi_buff = np.delete(csi_buff, delete_idxs, axis=0)
        n_ss = 1
        n_core = 1
        n_tot = n_ss * n_core
        start =0
        end = int(np.floor(csi_buff.shape[0]/n_tot))
        signal_complete = np.zeros((csi_buff.shape[1] - delete_idxs.shape[0], end-start, n_tot), dtype=complex)
        stream=0
        signal_stream = csi_buff[stream:end*n_tot + 1:n_tot, :][start:end, :]
        signal_stream = np.delete(signal_stream, delete_idxs, axis=1)
        mean_signal = np.mean(np.abs(signal_stream), axis=1, keepdims=True)
        H_m = signal_stream/mean_signal
        signal_complete[:, :, stream] = H_m.T
        with open (f"{save_path}/{i}/{j}.txt", 'wb') as f:
            pickle.dump(signal_complete, f)


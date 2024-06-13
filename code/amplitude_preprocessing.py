#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import Python library
import numpy as np
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


# In[2]:


# compute Amplitude and removing nan
# compute amplitude
def amplitude(df):
    amp = []
    d = np.array(df)
    for j in range(len(d)):
        imaginary = []
        real = []
        amplitudes = []
        for i in range(len(d[j])):
            if i % 2 == 0:
                imaginary.append(d[j][i])
            else:
                real.append(d[j][i])
        for i in range(int(len(d[0]) / 2)):
            amplitudes.append(sqrt(imaginary[i] ** 2 + real[i] ** 2))
        amp.append(amplitudes)
    amp = pd.DataFrame(amp)
    amp = amp.reset_index(drop=True)
    print("amp_phase completed")  
    return amp

# remove nan
def remove_nan(matrix_):
    temp = matrix_.copy()
    temp=temp.dropna()
    temp = temp.reset_index(drop=True)
    print("remove_nan completed")
    return  temp


# In[3]:


#denoising CSI data
#hampel filter
def hampel_filter(input_matrix, window_size, n_sigmas=3):
    # Perform Hampel filtering
    n_rows, n_cols = input_matrix.shape
    new_matrix = np.zeros_like(input_matrix)
    std_dev = np.std(input_matrix)
    mad = np.median(np.abs(input_matrix - np.median(input_matrix)))
    
    k = std_dev / (mad)

    for col_idx in range(n_cols):
        for ti in range(n_rows):
            start_idx = max(0, ti - window_size)
            end_idx = min(n_rows, ti + window_size)
            
            # Calculate the median of the window for the current column
            x0 = np.nanmedian(input_matrix[start_idx:end_idx, col_idx])
            
            # Calculate the median absolute deviation (MAD) of the window for the current column
            s0 = k * np.nanmedian(np.abs(input_matrix[start_idx:end_idx, col_idx] - x0))
            
            # Detect outliers based on the median and MAD
            if np.abs(input_matrix[ti, col_idx] - x0) > n_sigmas * s0:
                # Replace outliers with the median value
                new_matrix[ti, col_idx] = x0
            else:
                new_matrix[ti, col_idx] = input_matrix[ti, col_idx]
    print('hampel')
    return new_matrix

# 1-D wavelet transform
def denoise(df,wavelt, sigma):
    dwt = pd.DataFrame()
    # wevelt='bior1.1'
    for i in range(len(df.columns)):
        signal = df.iloc[:, i]
        # Perform wavelet decomposition
        coeff = pywt.wavedec(signal, wavelet=wavelt, mode="per")
        # Estimate noise level
        d = np.std(coeff[-1])
        sigma = 4 * d
        uthresh = sigma * np.sqrt(2 * np.log(len(signal)))

        # Apply thresholding to wavelet coefficients
        denoised_coeff = [coeff[0]]
        for c in coeff[1:]:
            denoised_coeff.append(pywt.threshold(c, value=uthresh, mode='soft'))

        # Reconstruct denoised signal
        denoised_signal = pywt.waverec(denoised_coeff, wavelet=wavelt, mode='per')
        # Store denoised signal in DataFrame
        dwt[i] = denoised_signal

    print("Denoising completed")
    return dwt

# savgol_filter
def smooth(df):
    from scipy.signal import savgol_filter
    window_length = 5  # Choose an appropriate window length (odd number)
    poly_order = 2  # Choose an appropriate polynomial order
    smoothed_data = savgol_filter(df, window_length, poly_order)
    smoothed_data= pd.DataFrame(smoothed_data)
    print('smooth')
    return smoothed_data


# In[ ]:


# computing 
save_path = os.path.abspath('./preprocessing/preprocessed_amp')
dir_path = os.path.abspath('./raw_csi') # raw data folder change the path as required 
dir_files= os.listdir(dir_path)
amp_files=[]
label_files=[]
for i in dir_files: 
    # print(i)
    files= os.listdir(f'{dir_path}/{i}')
    # print(files)
    for j in files:
        
        result = pd.read_csv(f'{dir_path}/{i}/{j}')
        len_= result.shape[0]
        len_=int(abs(0.03*len_))
        # print(len)
        result=result.iloc[len_:-len_, 128:] # remove first 128 rows. 
        
        label= pd.DataFrame(result['headlabel'])
        # label.to_csv(f'{save_path}/{i}_amp/label_{j}', index=False)
        result=result.drop(columns=['timestamp', 'headlabel'], axis=1)

        result.columns=range(len(result.columns))
        # remove null and pilot subcarriers
        
        delete_idxs =  np.asarray([0,1,2,3,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,254,255])
        
        result = result.drop(result.columns[delete_idxs], axis=1)
        result.columns=range(len(result.columns))
        amp_result= amplitude(result)
        print(amp_result.shape)
        amp_result= hampel_filter(np.asarray(amp_result), 1000)
        amp_result= denoise(pd.DataFrame(amp_result), 'db4', sigma=0.)
        amp_result= smooth(amp_result)
        amp_result= pd.concat([amp_result, label], axis=1)
        # amp_result= amp_result
        amp_result= remove_nan(amp_result)
        print(amp_result.isnull().any(axis=1).sum(), j)
        # print(amp_result.shape)
        amp_result.to_csv(f'{save_path}/{i}_amp/amp_{j}', index=False)
            
            
            


# In[ ]:





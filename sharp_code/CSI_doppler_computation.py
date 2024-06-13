import argparse
import numpy as np
import scipy.io as sio
import math as mt
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from scipy.signal.windows import hann
import pickle
import scipy.signal as signal
import pywt
import pandas as pd
import os
import sys


# input type is array
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
    new_matrix = pd.DataFrame(new_matrix)
    print("hampel filter done")
    return new_matrix
# input type is pandas

def denoise(df,wavelt, sigma):
    df_list=[]
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
        denoised_signal= np.nan_to_num(denoised_signal)
        df_list.append(pd.DataFrame(denoised_signal))
    dwt = pd.concat(df_list, axis=1)
    print("Denoising completed")
    return dwt

# input type is smooth. 
def smooth(df):
    from scipy.signal import savgol_filter
    window_length = 5  # Choose an appropriate window length (odd number)
    poly_order = 2  # Choose an appropriate polynomial order
    smoothed_data = savgol_filter(df, window_length, poly_order)
    smoothed_data= pd.DataFrame(smoothed_data)
    print('smooth complete')
    return smoothed_data

sliding = 1
noise_lev = -0.5

num_symbols = 31  # 51
middle = int(mt.floor(num_symbols / 2))
Tc = 6e-3
fc = 2.4e9 
v_light = 3e8
delta_v = round(v_light / (Tc * fc * num_symbols), 3)


main_dir = os.path.abspath('./processed')
save = os.path.abspath('./dopplers')
main_list= os.listdir(main_dir)
main_list.sort()

for i in main_list:
    sub_dir= f'{main_dir}/{i}'
    sub_list= os.listdir(sub_dir)
    sub_list.sort()
    for j in sub_list:
        os.makedirs(f'{save}/{i}/{j}', exist_ok=True)
        file_dir= f'{main_dir}/{i}/{j}'
        file_list=os.listdir(file_dir)
        for li in file_list:
            if os.path.exists(f'{save}/{i}/{j}/{li}'):
                print('already exist')
            else: 
                path = f'{main_dir}/{i}/{j}/{li}'
                with open(path, 'rb') as f:
                    csi_buff = pickle.load(f)
                csi_buff= csi_buff.reshape((len(csi_buff), -1))
                if csi_buff.shape[0] < num_symbols:
                    continue
                # csi_buff_ = csi_buff[:, :]
                print(csi_buff.shape)
                
                csi_buff = hampel_filter(csi_buff, 1000)
                csi_buff= csi_buff.dropna()
                    
                # print(csi_buff.isna())
                
                csi_buff= denoise(csi_buff, 'db4', sigma=0.1)
                print(csi_buff)
                try:
                    csi_buff= smooth(csi_buff)
                except:
                    pass
                print(csi_buff.isna())
                
                print(csi_buff.shape)
                csi_buff_= np.asarray(csi_buff)
                csi_matrix_processed= csi_buff_.reshape(csi_buff_.shape[0], 114, 2)
                csi_matrix_processed = csi_matrix_processed[1:-1, :, :]
                
                csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(csi_matrix_processed[:, :, 0],
                                                                                                    axis=1,  keepdims=True)
                csi_matrix_complete = csi_matrix_processed[:, :, 0]*np.exp(1j*csi_matrix_processed[:, :, 1])
                
                csi_d_profile_list = []
                for k in range(0, csi_matrix_complete.shape[0]-num_symbols, sliding):
                    csi_matrix_cut = csi_matrix_complete[k:k+num_symbols, :]
                    csi_matrix_cut = np.nan_to_num(csi_matrix_cut)

                    hann_window = np.expand_dims(hann(num_symbols), axis=-1)
                    csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
                # 31 X 114
                    
                    csi_doppler_prof = fft(csi_matrix_wind, n=100, axis=0)
                    # 100 X 114
                    csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)
                    
                    csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))
                    
                    csi_d_map = np.sum(csi_d_map, axis=1)
                
                    csi_d_profile_list.append(csi_d_map)
                csi_d_profile_array = np.asarray(csi_d_profile_list)
                csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
                # print(csi_d_profile_array_max)
                csi_d_profile_array = csi_d_profile_array/csi_d_profile_array_max
                csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)
                # print(csi_d_profile_array.shape)
                # print(np.isnan(csi_d_profile_array))
        
                name_file=f'{save}/{i}/{j}/{li}'
                print("name_file: ", name_file)
                with open(name_file, "wb") as fp:  # Pickling
                    pickle.dump(csi_d_profile_array, fp)

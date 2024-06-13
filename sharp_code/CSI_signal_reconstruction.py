import pickle
import math as mt
import numpy as np
from os import listdir, path
import sys
import os


main_dir = os.path.abspath('./preprocessing')
save = os.path.abspath('./processed')
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
            if li.startswith('Tr') and li.endswith('.txt'):
                path = f'{main_dir}/{i}/{j}/{li}'
                with open(path, 'rb') as fp:
                    H_est = pickle.load(fp)
                # print(H_est.shape)
                start_idx=0
                end_idx=-1
                end_H = H_est.shape[1]
                H_est = H_est[:, start_idx:end_H-end_idx]
                F_frequency = 114
                csi_matrix_processed = np.zeros((H_est.shape[1], F_frequency, 2))
                csi_matrix_processed[:, 0:-1, 0] = np.abs(H_est[0:-1, :]).T
                phase_before = np.unwrap(np.angle(H_est[0:-1, :]), axis=0)
                phase_err_tot = np.diff(phase_before, axis=1)
                ones_vector = np.ones((2, phase_before.shape[0]))
                ones_vector[1, :] = np.arange(0, phase_before.shape[0])

                for tidx in range(1, phase_before.shape[1]):
                    stop = False
                    idx_prec = -1
                    while not stop:
                        phase_err = phase_before[:, tidx] - phase_before[:, tidx - 1]
                        diff_phase_err = np.diff(phase_err)
                        idxs_invert_up = np.argwhere(diff_phase_err > 0.9 * mt.pi)[:, 0]
                        idxs_invert_down = np.argwhere(diff_phase_err < -0.9 * mt.pi)[:, 0]
                        if idxs_invert_up.shape[0] > 0:
                            idx_act = idxs_invert_up[0]
                            if idx_act == idx_prec:  # to avoid a continuous jump
                                stop = True
                            else:
                                phase_before[idx_act + 1:, tidx] = phase_before[idx_act + 1:, tidx] \
                                                                    - 2 * mt.pi
                                idx_prec = idx_act
                        elif idxs_invert_down.shape[0] > 0:
                            idx_act = idxs_invert_down[0]
                            if idx_act == idx_prec:
                                stop = True
                            else:
                                phase_before[idx_act + 1:, tidx] = phase_before[idx_act + 1:, tidx] \
                                                                    + 2 * mt.pi
                                idx_prec = idx_act
                        else:
                            stop = True
                for tidx in range(1, H_est.shape[1] - 1):
                    val_prec = phase_before[:, tidx - 1:tidx]
                    val_act = phase_before[:, tidx:tidx + 1]
                    error = val_act - val_prec
                    temp2 = np.linalg.lstsq(ones_vector.T, error)[0]
                    phase_before[:, tidx] = phase_before[:, tidx] - (np.dot(ones_vector.T, temp2)).T
                csi_matrix_processed[:, 0:-1, 1] = phase_before.T
                print(csi_matrix_processed.shape)
                temp_name= li.split('_')
                name_file= f'{save}/{i}/{j}/{temp_name[2]}'
                print("name_file: ", name_file)
                with open(name_file, "wb") as fp:  # Pickling
                    pickle.dump(csi_matrix_processed, fp)
                
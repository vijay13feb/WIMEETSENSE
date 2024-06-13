import numpy as np
import cmath as cmt
import osqp
import scipy
import argparse
import numpy as np
import scipy.io as sio
from os import listdir
import joblib
import pandas as pd
from os import path
import pickle
import sys
import os
# nohup python CSI_signal_Hestimation.py > hestimation.out 2>&1 &
# tail -f hestimation.out    to check the output
# ps aux | grep CSI_s
def convert_to_complex_osqp(real_im_n):
    len_vect = real_im_n.shape[0] // 2
    complex_n = real_im_n[:len_vect] + 1j * real_im_n[len_vect:]
    return complex_n

def lasso_regression_osqp_fast(H_matrix_, T_matrix_, selected_subcarriers, row_T, col_T, Im, Onm, P, q, A2, A3,
                               ones_n_matr, zeros_n_matr, zeros_nm_matr):
    # time_start = time.time()
    T_matrix_selected = T_matrix_[selected_subcarriers, :]
    H_matrix_selected = H_matrix_[selected_subcarriers]

    T_matrix_real = np.zeros((2*row_T, 2*col_T))
    T_matrix_real[:row_T, :col_T] = np.real(T_matrix_selected)
    T_matrix_real[row_T:, col_T:] = np.real(T_matrix_selected)
    T_matrix_real[row_T:, :col_T] = np.imag(T_matrix_selected)
    T_matrix_real[:row_T, col_T:] = - np.imag(T_matrix_selected)

    H_matrix_real = np.zeros((2*row_T))
    H_matrix_real[:row_T] = np.real(H_matrix_selected)
    H_matrix_real[row_T:] = np.imag(H_matrix_selected)

    n = col_T*2

    # OSQP data
    A = scipy.sparse.vstack([scipy.sparse.hstack([T_matrix_real, -Im, Onm.T]),
                             A2,
                             A3], format='csc')
    l = np.hstack([H_matrix_real, - np.inf * ones_n_matr, zeros_n_matr])
    u = np.hstack([H_matrix_real, zeros_n_matr, np.inf * ones_n_matr])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

    # Update linear cost
    lambd = 1E-1
    q_new = np.hstack([zeros_nm_matr, lambd * ones_n_matr])
    prob.update(q=q_new)

    # Solve
    res = prob.solve()

    x_out = res.x
    x_out_cut = x_out[:n]

    r_opt = convert_to_complex_osqp(x_out_cut)
    return r_opt

def build_T_matrix(frequency_vector, delta_t_, t_min_, t_max_):
    F_frequency = frequency_vector.shape[0]
    L_paths = int((t_max_ - t_min_) / delta_t_)
    T_matrix = np.zeros((F_frequency, L_paths), dtype=complex)
    time_matrix = np.zeros((L_paths,))
    for col in range(L_paths):
        time_col = t_min_ + delta_t_ * col
        time_matrix[col] = time_col
        for row in range(F_frequency):
            freq_n = frequency_vector[row]
            T_matrix[row, col] = cmt.exp(-1j * 2 * cmt.pi * freq_n * time_col)
    return T_matrix, time_matrix

main_dir = os.path.abspath('./preprocessing')

main_list= os.listdir(main_dir)
main_list.sort()

for i in main_list:
    sub_dir= f'{main_dir}/{i}'
    sub_list= os.listdir(sub_dir)
    sub_list.sort()
    for j in sub_list:
        file_dir= f'{main_dir}/{i}/{j}'
        file_list=os.listdir(file_dir)
        for k in file_list:
            if k.startswith("Tr_vector"):
                print("alreday exist")
            else: 
                path = f'{main_dir}/{i}/{j}/{k}'
                print(path)
                with open(path, 'rb') as fp:
                    signal_complete = pickle.load(fp)

                print(signal_complete.shape)

                subcarriers_space = 2
                delta_t = 1E-7
                delta_t_refined = 2.4E-9
                range_refined_up = 2.457E-7
                range_refined_down = 2.417E-7
                end_r =-1
                start_r = 0
                if end_r != -1:
                    end_r = end_r
                else:
                    end_r = signal_complete.shape[1]
                F_frequency = 114
                delta_f = 312.5E3
                frequency_vector_complete = np.zeros(F_frequency, )
                delete_idxs=[]
                F_frequency_2 = F_frequency // 2
                for row in range(F_frequency_2):
                    freq_n = delta_f * (row - F_frequency / 2)
                    frequency_vector_complete[row] = freq_n
                    freq_p = delta_f * row
                    frequency_vector_complete[row + F_frequency_2] = freq_p
                frequency_vector = np.delete(frequency_vector_complete, delete_idxs)

                T = 1/delta_f
                t_min = -3E-7
                t_max = 5E-7

                T_matrix, time_matrix = build_T_matrix(frequency_vector, delta_t, t_min, t_max)
                r_length = int((t_max - t_min) / delta_t_refined)

                start_subcarrier = 0
                end_subcarrier = frequency_vector.shape[0]
                select_subcarriers = np.arange(start_subcarrier, end_subcarrier, subcarriers_space)

                start_subcarrier = 0
                end_subcarrier = frequency_vector.shape[0]
                select_subcarriers = np.arange(start_subcarrier, end_subcarrier, subcarriers_space)

                n_ss = 1
                n_core = 1
                n_tot = n_ss * n_core

                row_T = int(T_matrix.shape[0] / subcarriers_space)
                col_T = T_matrix.shape[1]
                m = 2 * row_T
                n = 2 * col_T
                In = scipy.sparse.eye(n)
                Im = scipy.sparse.eye(m)
                On = scipy.sparse.csc_matrix((n, n))
                Onm = scipy.sparse.csc_matrix((n, m))
                P = scipy.sparse.block_diag([On, Im, On], format='csc')
                q = np.zeros(2 * n + m)
                A2 = scipy.sparse.hstack([In, Onm, -In])
                A3 = scipy.sparse.hstack([In, Onm, In])
                ones_n_matr = np.ones(n)
                zeros_n_matr = np.zeros(n)
                zeros_nm_matr = np.zeros(n + m)
                stream =0
                signal_considered = signal_complete[:, start_r:end_r, stream]
                r_optim = np.zeros((r_length, end_r - start_r), dtype=complex)
                Tr_matrix = np.zeros((frequency_vector_complete.shape[0], end_r - start_r), dtype=complex)

                for time_step in range(end_r - start_r):
                    signal_time = signal_considered[:, time_step]
                    complex_opt_r = lasso_regression_osqp_fast(signal_time, T_matrix, select_subcarriers, row_T, col_T,
                                                                Im, Onm, P, q, A2, A3, ones_n_matr, zeros_n_matr,
                                                                zeros_nm_matr)

                    position_max_r = np.argmax(abs(complex_opt_r))
                    time_max_r = time_matrix[position_max_r]

                    T_matrix_refined, time_matrix_refined = build_T_matrix(frequency_vector, delta_t_refined,
                                        # joblib.dump(Tr_matrix,f'{path_file}/Tr_vector_{file_name}')
                                                       max(time_max_r - range_refined_down, t_min),
                                                                            min(time_max_r + range_refined_up, t_max))

                    # Auxiliary data for second step
                    col_T_refined = T_matrix_refined.shape[1]
                    n_refined = 2 * col_T_refined
                    In_refined = scipy.sparse.eye(n_refined)
                    On_refined = scipy.sparse.csc_matrix((n_refined, n_refined))
                    Onm_refined = scipy.sparse.csc_matrix((n_refined, m))
                    P_refined = scipy.sparse.block_diag([On_refined, Im, On_refined], format='csc')
                    q_refined = np.zeros(2 * n_refined + m)
                    A2_refined = scipy.sparse.hstack([In_refined, Onm_refined, -In_refined])
                    A3_refined = scipy.sparse.hstack([In_refined, Onm_refined, In_refined])
                    ones_n_matr_refined = np.ones(n_refined)
                    zeros_n_matr_refined = np.zeros(n_refined)
                    zeros_nm_matr_refined = np.zeros(n_refined + m)

                    complex_opt_r_refined = lasso_regression_osqp_fast(signal_time, T_matrix_refined, select_subcarriers,
                                                                                row_T, col_T_refined, Im, Onm_refined, P_refined,
                                                                                q_refined, A2_refined, A3_refined,
                                                                                ones_n_matr_refined, zeros_n_matr_refined,
                                                                                zeros_nm_matr_refined)

                    position_max_r_refined = np.argmax(abs(complex_opt_r_refined))

                    T_matrix_refined, time_matrix_refined = build_T_matrix(frequency_vector_complete, delta_t_refined,
                                                                            max(time_max_r - range_refined_down, t_min),
                                                                            min(time_max_r + range_refined_up, t_max))

                    Tr = np.multiply(T_matrix_refined, complex_opt_r_refined)

                    Tr_sum = np.sum(Tr, axis=1)

                    Trr = np.multiply(Tr, np.conj(Tr[:, position_max_r_refined:position_max_r_refined + 1]))
                    Trr_sum = np.sum(Trr, axis=1)

                    Tr_matrix[:, time_step] = Trr_sum
                    time_max_r = time_matrix_refined[position_max_r_refined]

                    start_r_opt = int((time_matrix_refined[0] - t_min)/delta_t_refined)
                    end_r_opt = start_r_opt + complex_opt_r_refined.shape[0]
                    r_optim[start_r_opt:end_r_opt, time_step] = complex_opt_r_refined

                # name_file=f'{path_file}/Tr_vector_{file_name}'
                save_path = f'{main_dir}/{i}/{j}/Tr_vector_{k[:-4]}.txt'
           
                with open(save_path, "wb") as fp:  # Pickling
                    pickle.dump(Tr_matrix, fp)
                save_path = f'{main_dir}/{i}/{j}/r_vector_{k[:-4]}.txt'
                with open(save_path, "wb") as fp:  # Pickling
                    pickle.dump(r_optim, fp)

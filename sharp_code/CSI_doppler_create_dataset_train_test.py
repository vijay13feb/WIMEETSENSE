#python CSI_doppler_create_dataset_train_test.py ./ S4 31 1 340 30 Fo,LD,LL,LR,LU,No,Sh 1

import argparse
import glob
import os
import sys
import numpy as np
import pickle
import math as mt
import shutil
from dataset_utility import create_windows_antennas, convert_to_number


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')  # ./doppler_traces/
    parser.add_argument('subdirs', help='Sub-directories')  # S1a
    parser.add_argument(
        'sample_lengths', help='Number of packets in a sample', type=int)  # 31
    parser.add_argument(
        'sliding', help='Number of packet for sliding operations', type=int)  # 1
    parser.add_argument(
        'window_length', help='Number of samples per window', type=int)  # 340
    parser.add_argument(
        'stride_length', help='Number of samples to stride', type=int)  # 30
    # label activities.
    parser.add_argument('labels_activities',
                        help='Labels of the activities to be considered')
    parser.add_argument(
        'n_tot', help='Number of streams * number of antennas', type=int)  # 1
    # parser.add_argument('antenna_number', help='antenna index') # 1
    # parser.add_argument(
    #     'start_with', help='start_with the file name')
    args = parser.parse_args()

    labels_activities = args.labels_activities  # list of activities  A,B,C,D,F

    csi_label_dict = []  # ['A', 'B', 'C', 'D', 'F']
    for lab_act in labels_activities.split(','):
        csi_label_dict.append(lab_act)
    print(csi_label_dict)

    activities = np.asarray(labels_activities)  # A,B,C,D,F

    n_tot = args.n_tot  # 4
    num_packets = args.sample_lengths  # 31
    middle = int(np.floor(num_packets / 2))  # 15
    list_subdir = args.subdirs  # string S1a,S1b,S1c

    for subdir in list_subdir.split(','):
        print(subdir)
        exp_dir = args.dir + subdir + '/'  # ./doppler_traces/S1a/

        path_train = exp_dir + 'train_antennas_' + str(activities)
        path_val = exp_dir + 'val_antennas_' + str(activities)
        path_test = exp_dir + 'test_antennas_' + str(activities)
        paths = [path_train, path_val, path_test]
        # ['./doppler_traces/S1a/train_antennas_A,B,C,D,F', './doppler_traces/S1a/val_antennas_A,B,C,D,F', './doppler_traces/S1a/test_antennas_A,B,C,D,F']

        for pat in paths:
            if os.path.exists(pat):
                remove_files = glob.glob(pat + '/*')
                for f in remove_files:  # make directories mentioned above
                    os.remove(f)
            else:
                os.mkdir(pat)

        # ./doppler_traces/S1a/complete_antennas_A,B,C,D,F
        path_complete = exp_dir + 'complete_antennas_' + str(activities)
        if os.path.exists(path_complete):
            shutil.rmtree(path_complete)
            pass

        names = []
        all_files = os.listdir(exp_dir)  # ./doppler_traces/S1a/
        #all_files= [d for d in all_files if d.startswith(args.start_with)]
        print(all_files)
        for i in all_files:
            if "Forward" in i or "Looking" in i or "Nodding" in i or "Shaking" in i:
                names.append(i[:-4])
        names.sort()  # all files like S1a_F_stream_1.txt sorts all the files
        print("names are: ", names)
        # sys.exit(0)
        csi_matrices = []
        labels = []
        lengths = []
        label = 'null'
        prev_label = label
        csi_matrix = []
        processed = False

        for i_name, name in enumerate(names):

            if i_name % n_tot == 0 and i_name != 0 and processed:
                ll = csi_matrix[0].shape[1]  # total rows 11169

                for i_ant in range(1, n_tot):
                    if ll != csi_matrix[i_ant].shape[1]:
                        break
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                csi_matrix = []
            label = csi_label_dict[i_name]
            print("label", label)

            if label not in csi_label_dict:
                print('victim')
                processed = False
                continue
            processed = True

            # convert activites name in number such as Forward as 0.
            label = convert_to_number(label, csi_label_dict)

            print("label_number: ", label)
            if i_name % n_tot == 0:

                prev_label = label  # add label number here
            elif label != prev_label:
                print('error in ' + str(name))
                break
            name_file = exp_dir + name + '.txt'
            # name_file = exp_dir + name + '.pkl'
            with open(name_file, "rb") as fp:  # Unpickling
                stft_sum_1 = pickle.load(fp)  # read all the activties files in the
            # print(stft_sum_1.shape)

            stft_sum_1_mean = stft_sum_1 - \
                np.mean(stft_sum_1, axis=0, keepdims=True)

            csi_matrix.append(stft_sum_1_mean.T)  # 100 X 11169
        error = False
        if processed:
            # for the last block
            if len(csi_matrix) < n_tot:
                print('error in ' + str(name))
            ll = csi_matrix[0].shape[1]
            # print(ll)

            for i_ant in range(1, n_tot):
                if ll != csi_matrix[i_ant].shape[1]:
                    print('error in ' + str(name))
                    error = True
            if not error:
                lengths.append(ll)
                csi_matrices.append(np.asarray(csi_matrix))
                labels.append(label)
                # [0, 1, 2, 3, 4, 5, 6]
        if not error:
            lengths = np.asarray(lengths)
            length_min = np.min(lengths)
            print('lengths', lengths)
            csi_train = []
            csi_val = []
            csi_test = []
            length_train = []
            length_val = []
            length_test = []
            # print("lables", labels)
            for i in range(len(labels)):
                ll = lengths[i]  # 11169
                train_len = int(np.floor(ll * 0.6))  # train length
                print(train_len)
                length_train.append(train_len)

                csi_train.append(csi_matrices[i][:, :, :train_len])

                start_val = train_len + mt.ceil(num_packets/1)

                val_len = int(np.floor(ll * 0.2))
                # print(val_len)
                length_val.append(val_len)
                csi_val.append(
                    csi_matrices[i][:, :, start_val:start_val + val_len])

                start_test = start_val + val_len + mt.ceil(num_packets/1)
                length_test.append(ll - val_len - train_len -
                                2*mt.ceil(num_packets/1))
                csi_test.append(csi_matrices[i][:, :, start_test:])
            print("length_train: ", length_train)
            print("sum: ", sum(length_train))
            print("length_val: ", length_val)
            print("sum: ", sum(length_val))
            print("length_test: ", length_test)
            print("sum: ", sum(length_test))
            # exit(0)

            list_sets_name = ['train', 'val', 'test']
            list_sets = [csi_train, csi_val, csi_test]
            list_sets_lengths = [length_train, length_val, length_test]

            for set_idx in range(3):
                csi_matrices_set, labels_set = create_windows_antennas(list_sets[set_idx], labels, args.window_length,
                                                                    args.stride_length, remove_mean=False)
                print(len(labels_set), 'label')

                num_windows = np.floor((np.asarray(
                    list_sets_lengths[set_idx]) - args.window_length) / args.stride_length + 1)

                names_set = []
                suffix = '.txt'
                for ii in range(len(csi_matrices_set)):
                    name_file = exp_dir + list_sets_name[set_idx] + '_antennas_' + str(activities) + '/' + \
                        str(ii) + suffix
                    names_set.append(name_file)
                    # print("name_file: ", name_file)
                    with open(name_file, "wb") as fp:  # Pickling
                        pickle.dump(csi_matrices_set[ii], fp)
                    # print("len(csi_matrices_set[ii]): ", len(csi_matrices_set[ii]))

                name_labels = exp_dir + '/' + '/labels_' + \
                    list_sets_name[set_idx] + '_antennas_' + \
                    str(activities) + suffix
                with open(name_labels, "wb") as fp:  # Pickling
                    pickle.dump(labels_set, fp)

                name_f = exp_dir + '/' + '/files_' + \
                    list_sets_name[set_idx] + '_antennas_' + \
                    str(activities) + suffix
                print("name_f: ", name_f)
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(names_set, fp)
                print("len(names_set) : ", len(names_set))

                name_f = exp_dir + '/' + '/num_windows_' + \
                    list_sets_name[set_idx] + '_antennas_' + \
                    str(activities) + suffix
                print("name_f: ", name_f)
                with open(name_f, "wb") as fp:  # Pickling
                    pickle.dump(num_windows, fp)

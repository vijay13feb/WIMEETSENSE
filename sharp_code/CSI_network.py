#python CSI_network.py ./ S4 100 340 1 16  1 temp  Fo,LD,LL,LR,LU,No,Sh
"""
    Copyright (C) 2022 Francesca Meneghello
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import os
import sys
from dataset_utility import create_dataset_single, expand_antennas
from network_utility import *
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dir', help='Directory of data')
    parser.add_argument('subdirs', help='Subdirs for training')  # S1a
    parser.add_argument(
        # 100
        'feature_length', help='Length along the feature dimension (height)', type=int)
    parser.add_argument(
        # 340
        'sample_length', help='Length along the time dimension (width)', type=int)
    parser.add_argument('channels', help='Number of channels', type=int)  # 1
    parser.add_argument(
        'batch_size', help='Number of samples in a batch', type=int)  # 32
    parser.add_argument(
        'num_tot', help='Number of antenna * number of spatial streams', type=int)  # 4*1= 4
    # single_antennas
    parser.add_argument('name_base', help='Name base for the files')
    # activities #F,LD,LL,LR,LU,N,S
    parser.add_argument('activities', help='Activities to be considered')
    parser.add_argument('--bandwidth', help='Bandwidth in [MHz] to select the subcarriers, can be 20, 40, 80 '
                                            '(default 80)', default=80, required=False, type=int)
    parser.add_argument('--sub_band', help='Sub_band idx in [1, 2, 3, 4] for 20 MHz, [1, 2] for 40 MHz '
                                           '(default 1)', default=1, required=False, type=int)
    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    # print(gpus)

    bandwidth = args.bandwidth
    sub_band = args.sub_band

    csi_act = args.activities
    activities = []
    for lab_act in csi_act.split(','):
        activities.append(lab_act)
    activities = np.asarray(activities)
    print(activities.shape[0])
    name_base = args.name_base
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_train.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) +
                  '_cache_train.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_train.index')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_val.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) +
                  '_cache_val.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_val.index')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_train_test.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) +
                  '_cache_train_test.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_train_test.index')
    if os.path.exists(name_base + '_' + str(csi_act) + '_cache_test.data-00000-of-00001'):
        os.remove(name_base + '_' + str(csi_act) +
                  '_cache_test.data-00000-of-00001')
        os.remove(name_base + '_' + str(csi_act) + '_cache_test.index')

    subdirs_training = args.subdirs  # string
    labels_train = []
    all_files_train = []
    labels_val = []
    all_files_val = []
    labels_test = []
    all_files_test = []
    sample_length = args.sample_length  # 340
    feature_length = args.feature_length  # 100
    channels = args.channels  # 1
    num_antennas = args.num_tot  # 4
    # input shape for the model  training num of antennas X sample length X feature length X channels
    # 4 340 100 1
    input_shape = (num_antennas, sample_length, feature_length, channels)
    # input network sample length 340, featur length 100 and channels 1
    input_network = (sample_length, feature_length, channels)
    # batch size is 32
    batch_size = args.batch_size
    # ouput shapes is number of activities to be detected.
    output_shape = activities.shape[0]
    # labels considered.
    labels_considered = np.arange(output_shape)
    # activities
    activities = activities[labels_considered]

    suffix = '.txt'
    # suffix = '.pkl'

    for sdir in subdirs_training.split(','):
        exp_save_dir = args.dir + sdir + '/'
        dir_train = args.dir + sdir + '/train_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + \
            '/labels_train_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_train.extend(pickle.load(fp))
        name_f = args.dir + sdir + \
            '/files_train_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_train.extend(pickle.load(fp))

        dir_val = args.dir + sdir + '/val_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + \
            '/labels_val_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_val.extend(pickle.load(fp))
        name_f = args.dir + sdir + \
            '/files_val_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_val.extend(pickle.load(fp))

        dir_test = args.dir + sdir + '/test_antennas_' + str(csi_act) + '/'
        name_labels = args.dir + sdir + \
            '/labels_test_antennas_' + str(csi_act) + suffix
        with open(name_labels, "rb") as fp:  # Unpickling
            labels_test.extend(pickle.load(fp))
        name_f = args.dir + sdir + \
            '/files_test_antennas_' + str(csi_act) + suffix
        with open(name_f, "rb") as fp:  # Unpickling
            all_files_test.extend(pickle.load(fp))

    file_train_selected = [all_files_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                           labels_considered]
    # print(file_train_selected)
    labels_train_selected = [labels_train[idx] for idx in range(len(labels_train)) if labels_train[idx] in
                             labels_considered]

    file_train_selected_expanded, labels_train_selected_expanded, stream_ant_train = \
        expand_antennas(file_train_selected,
                        labels_train_selected, num_antennas)
    print("len(file_train_selected): ",len(file_train_selected))
    print("len(file_train_selected_expanded): ",len(file_train_selected_expanded))
    # print(len(stream_ant_train))

    name_cache = name_base + '_' + str(csi_act) + '_cache_train'
    dataset_csi_train = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                              stream_ant_train, input_network, batch_size,
                                              shuffle=True, cache_file=name_cache)
    # sys.exit()

    file_val_selected = [all_files_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                         labels_considered]
    labels_val_selected = [labels_val[idx] for idx in range(len(labels_val)) if labels_val[idx] in
                           labels_considered]

    file_val_selected_expanded, labels_val_selected_expanded, stream_ant_val = \
        expand_antennas(file_val_selected, labels_val_selected, num_antennas)

    print("len(file_val_selected): ",len(file_val_selected))
    print(len(file_val_selected_expanded))

    name_cache_val = name_base + '_' + str(csi_act) + '_cache_val'
    dataset_csi_val = create_dataset_single(file_val_selected_expanded, labels_val_selected_expanded,
                                            stream_ant_val, input_network, batch_size,
                                            shuffle=False, cache_file=name_cache_val)

    file_test_selected = [all_files_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                          labels_considered]
    labels_test_selected = [labels_test[idx] for idx in range(len(labels_test)) if labels_test[idx] in
                            labels_considered]

    file_test_selected_expanded, labels_test_selected_expanded, stream_ant_test = \
        expand_antennas(file_test_selected, labels_test_selected, num_antennas)

    print("len(file_test_selected): ",len(file_test_selected))
    print("len(file_test_selected_expanded): ",len(file_test_selected_expanded))

    name_cache_test = name_base + '_' + str(csi_act) + '_cache_test'
    dataset_csi_test = create_dataset_single(file_test_selected_expanded, labels_test_selected_expanded,
                                             stream_ant_test, input_network, batch_size,
                                             shuffle=False, cache_file=name_cache_test)
  # csi model takes input shape and output shape.

    # sys.exit()
    tf.random.set_seed(42)
    np.random.seed(42)
    csi_model = csi_network_inc_res(input_network, output_shape)
    

    optimiz = tf.keras.optimizers.Adam(learning_rate=0.0001)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='True')
    csi_model.compile(optimizer=optimiz, loss=loss, metrics=[
                      tf.keras.metrics.SparseCategoricalAccuracy()])

    # training data sampled
    num_samples_train = len(file_train_selected_expanded)
    print(num_samples_train)
    # validation data sampled
    num_samples_val = len(file_val_selected_expanded)
    print(num_samples_val)
    num_samples_test = len(file_test_selected_expanded)  # test data
    print(num_samples_test)
    lab, count = np.unique(labels_train_selected_expanded, return_counts=True)

    lab_val, count_val = np.unique(
        labels_val_selected_expanded, return_counts=True)
    lab_test, count_test = np.unique(
        labels_test_selected_expanded, return_counts=True)
    train_steps_per_epoch = int(np.ceil(num_samples_train/batch_size))
    val_steps_per_epoch = int(np.ceil(num_samples_val/batch_size))
    test_steps_per_epoch = int(np.ceil(num_samples_test/batch_size))

    callback_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy', patience=20)

    name_model = "./model/"+str(subdirs_training)+"_"+str(name_base)+"_"+"network.h5"
    
  
    
    callback_save = tf.keras.callbacks.ModelCheckpoint(name_model, save_freq='epoch', save_best_only=True,
                                                       monitor='val_sparse_categorical_accuracy')

    results = csi_model.fit(dataset_csi_train, epochs=20, steps_per_epoch=train_steps_per_epoch,
                            validation_data=dataset_csi_val, validation_steps=val_steps_per_epoch,
                            callbacks=[callback_save, callback_stop])

    with tf.keras.utils.custom_object_scope({'AttenLayer': AttenLayer}):
        csi_model = tf.keras.models.load_model(name_model)
    csi_model.summary()
    # csi_model = tf.keras.models.load_model(name_model)

    # train
    train_labels_true = np.array(labels_train_selected_expanded)

    name_cache_train_test = name_base + '_' + \
        str(csi_act) + '_cache_train_test'
    dataset_csi_train_test = create_dataset_single(file_train_selected_expanded, labels_train_selected_expanded,
                                                   stream_ant_train, input_network, batch_size,
                                                   shuffle=False, cache_file=name_cache_train_test, prefetch=False)
    train_prediction_list = csi_model.predict(dataset_csi_train_test,
                                              steps=train_steps_per_epoch)[:train_labels_true.shape[0]]

    train_labels_pred = np.argmax(train_prediction_list, axis=1)

    conf_matrix_train = confusion_matrix(train_labels_true, train_labels_pred)

    # val
    val_labels_true = np.array(labels_val_selected_expanded)
    val_prediction_list = csi_model.predict(
        dataset_csi_val, steps=val_steps_per_epoch)[:val_labels_true.shape[0]]

    val_labels_pred = np.argmax(val_prediction_list, axis=1)

    conf_matrix_val = confusion_matrix(val_labels_true, val_labels_pred)

    # test
    test_labels_true = np.array(labels_test_selected_expanded)

    test_prediction_list = csi_model.predict(dataset_csi_test, steps=test_steps_per_epoch)[
        :test_labels_true.shape[0]]

    test_labels_pred = np.argmax(test_prediction_list, axis=1)

    conf_matrix = confusion_matrix(test_labels_true, test_labels_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(test_labels_true,
                                                                   test_labels_pred,
                                                                   labels=labels_considered)
    accuracy = accuracy_score(test_labels_true, test_labels_pred)

    print(accuracy)
    print(fscore)

    # merge antennas test
    labels_true_merge = np.array(labels_test_selected)
    pred_max_merge = np.zeros_like(labels_test_selected)
    for i_lab in range(len(labels_test_selected)):
        pred_antennas = test_prediction_list[i_lab *
                                             num_antennas:(i_lab + 1) * num_antennas, :]
        lab_merge_max = np.argmax(np.sum(pred_antennas, axis=0))

        pred_max_antennas = test_labels_pred[i_lab *
                                             num_antennas:(i_lab + 1) * num_antennas]
        lab_unique, count = np.unique(pred_max_antennas, return_counts=True)
        lab_max_merge = -1
        if lab_unique.shape[0] > 1:
            count_argsort = np.flip(np.argsort(count))
            count_sort = count[count_argsort]
            lab_unique_sort = lab_unique[count_argsort]
            # ex aequo between two labels
            if count_sort[0] == count_sort[1] or lab_unique.shape[0] > 2:
                lab_max_merge = lab_merge_max
            else:
                lab_max_merge = lab_unique_sort[0]
        else:
            lab_max_merge = lab_unique[0]
        pred_max_merge[i_lab] = lab_max_merge

    conf_matrix_max_merge = confusion_matrix(
        labels_true_merge, pred_max_merge, labels=labels_considered)
    precision_max_merge, recall_max_merge, fscore_max_merge, _ = \
        precision_recall_fscore_support(
            labels_true_merge, pred_max_merge, labels=labels_considered, zero_division=1)
    accuracy_max_merge = accuracy_score(labels_true_merge, pred_max_merge)
    
    print("Max Accuracy: ", accuracy_max_merge)

    metrics_matrix_dict = {'conf_matrix': conf_matrix,
                        'accuracy_single': accuracy,
                        'precision_single': precision,
                        'recall_single': recall, 
                        'fscore_single': fscore,
                        'conf_matrix_max_merge': conf_matrix_max_merge,
                        'accuracy_max_merge': accuracy_max_merge,
                        'precision_max_merge': precision_max_merge,
                        'recall_max_merge': recall_max_merge,
                        'fscore_max_merge': fscore_max_merge}
print('F1-score, fscore')

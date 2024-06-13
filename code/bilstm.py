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
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import math
import difflib
import collections
import os
import shutil
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense, Concatenate
from tensorflow.keras.models import Model
import sys

# In[2]:


#loading the dataset
import sys
folder =sys.argv[1] # change the folder as per requirement
feat_train_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_train.csv'))
label_train_amp=pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_train.csv'))
feat_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_test.csv'))
label_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_test.csv'))


# In[3]:


# select 100 best
selector = SelectKBest(f_classif, k=100)
feat_train_amp_100 = selector.fit_transform(feat_train_amp, label_train_amp)
feat_test_amp_100 = selector.fit_transform(feat_test_amp, label_test_amp)


# In[4]:


# convert label using one hot encoding 
label_train_amp = pd.DataFrame(label_train_amp)
onehot_encoder = OneHotEncoder()
label_train_amp_one= onehot_encoder.fit_transform(label_train_amp)
label_test_amp_one= onehot_encoder.fit_transform(label_test_amp)
label_train_amp_one = label_train_amp_one.toarray()
label_test_amp_one = label_test_amp_one.toarray()


# In[5]:


# reshaoe input dimension
feat_train_amp_10X10=np.reshape(feat_train_amp_100,(feat_train_amp_100.shape[0],10,10))
feat_test_amp_10X10=np.reshape(feat_test_amp_100,(feat_test_amp_100.shape[0],10,10))


# In[6]:


# split dataset into train test and validation.

X_train, X_val, y_train, y_val=train_test_split(feat_train_amp_10X10, label_train_amp_one, test_size=0.20, random_state=42)


# In[7]:


# defining Attention layer.
class AttenLayer(tf.keras.layers.Layer):

    def __init__(self, num_state, **kw):
        super(AttenLayer, self).__init__(**kw)
        self.num_state = num_state

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel', shape=[input_shape[-1], self.num_state])
        self.bias = self.add_weight('bias', shape=[self.num_state])
        self.prob_kernel = self.add_weight('prob_kernel', shape=[self.num_state])

    def call(self, input_tensor):
        atten_state = tf.tanh(tf.tensordot(input_tensor, self.kernel, axes=1) + self.bias)
        logits = tf.tensordot(atten_state, self.prob_kernel, axes=1)
        prob = tf.nn.softmax(logits)
        weighted_feature = tf.reduce_sum(tf.multiply(input_tensor, tf.expand_dims(prob, -1)), axis=1)
        return weighted_feature

    # for saving the model
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_state': self.num_state,})
        return config


# In[8]:


# define Bi-LSTM model
class CSIModelConfig:
    def __init__(self, win_len=1000, step=200, thrshd=0.6, downsample=2):
        self._win_len = win_len
        self._step = step
        self._thrshd = thrshd
        self._labels = ("Forward", "Looking Down", "Looking Up", "Looking Left", "Looking Right", "Nodding", "Shaking")
        self._downsample = downsample

    def build_model(self, n_unit_lstm=200, n_unit_atten=400,l1_reg=0.01):
        """
        Returns the Tensorflow Model which uses AttenLayer
        """
        if self._downsample > 1:
            length = len(np.ones((self._win_len,))[::self._downsample])
            x_in = tf.keras.Input(shape=(length, 10))
        else:
            x_in = tf.keras.Input(shape=(self._win_len, 10))
        
        x_tensor = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=n_unit_lstm, return_sequences=True,kernel_regularizer=tf.keras.regularizers.l1(l1_reg)))(x_in)
        x_tensor = AttenLayer(n_unit_atten)(x_tensor)
        pred = tf.keras.layers.Dense(len(self._labels), activation='softmax')(x_tensor)
        model = tf.keras.Model(inputs=x_in, outputs=pred)
        return model


# In[10]:


cfg = CSIModelConfig(win_len=10, step=250, thrshd=0.8, downsample=1)
model = cfg.build_model(n_unit_lstm=400, n_unit_atten=400, l1_reg=0.0)
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
model.summary()
model.fit(
        X_train,
        y_train,
        batch_size=56, epochs=30,
        validation_data=[X_val, y_val],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(os.path.abspath(f'./model/bilstm_{folder}.sav'),
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                save_weights_only=False)
            ]
)


# In[11]:


from tensorflow.keras.models import load_model
model_path = os.path.abspath(f'./model/bilstm_{folder}.sav')
loaded_model = load_model(model_path)


# In[12]:


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
y_pred= loaded_model.predict(feat_test_amp_10X10)
y_predicted= np.argmax(y_pred, axis=1)
ground_truth= np.argmax(label_test_amp_one, axis=1)
re= f1_score(y_predicted, ground_truth,average='micro')
print("F1-score", re)
print("Accuracy", accuracy_score(y_predicted, ground_truth))


# In[ ]:





# 10 -fold cross validation

# In[ ]:


from sklearn.model_selection import KFold
kf=KFold(n_splits=10, random_state=None, shuffle=False)
accuracy_scores = []
f1_scores = []

cfg = CSIModelConfig(win_len=10, step=250, thrshd=0.8, downsample=1)
model = cfg.build_model(n_unit_lstm=400, n_unit_atten=400, l1_reg=0.0)

model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss='categorical_crossentropy', 
        metrics=['accuracy'])
model.summary()
for train_index, test_index in kf.split(X_train):
    X_train_, X_valid_ = X_train[train_index], X_train[test_index]
    Y_train_, Y_valid_ = y_train[train_index], y_train[test_index]
    model.fit(
            X_train_,
            Y_train_,
            batch_size=56, epochs=30,
            validation_data=(X_valid_, Y_valid_),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint('best_atten.hdf5',
                                                    monitor='val_accuracy',
                                                    save_best_only=True,
                                                    save_weights_only=False)
                ])  
    y_pred = model.predict(X_valid_)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(np.argmax(Y_valid_, axis=1), y_pred_classes)
    f1 = f1_score(np.argmax(Y_valid_, axis=1), y_pred_classes, average='weighted')

    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    print(f1_scores)

print("Average Accuracy:", np.mean(accuracy_scores))
print("Average F1 Score:", np.mean(f1_scores))



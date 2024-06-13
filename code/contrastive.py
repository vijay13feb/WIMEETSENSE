#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score


# In[8]:


#loading the dataset
import sys
folder =sys.argv[1] # change the folder as per requirement
feat_train_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_train.csv'))
label_train_amp=pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_train.csv'))
feat_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_test.csv'))
label_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_test.csv'))


# In[10]:


#preprocess step 
df = pd.concat([feat_train_amp, label_train_amp], axis=1)
feature_columns=[]
for i in range(114):
    feature_columns.append(i)
input=np.array(df.iloc[:,feature_columns])
target=np.array(df['headlabel'])

input=np.abs(input)

input=SelectKBest(chi2, k=96).fit_transform(input, target)

headlabels=np.unique(target)

dfs={}

for i in headlabels:
    dfs[i]=[]
for i in range(target.shape[0]):
    dfs[target[i]].append(input[i])


# In[11]:


import random

X_train=[0]*9
X_test=[0]*9
Y_train=[0]*9
Y_test=[0]*9

for i in range(7):
    random.shuffle(dfs[headlabels[i]])
    rows_to_rem=0-(len(dfs[headlabels[i]])%32)
    if(rows_to_rem!=0):
        dfs[headlabels[i]]=dfs[headlabels[i]][:rows_to_rem]
    rows=int(len(dfs[headlabels[i]])/32)
    dfs[headlabels[i]]=np.array(dfs[headlabels[i]])
    print(dfs[headlabels[i]].shape)
    dfs[headlabels[i]]=dfs[headlabels[i]].reshape(rows,32,32,3)
    X_train[i], X_test[i], Y_train[i], Y_test[i] = train_test_split(dfs[headlabels[i]], np.array([headlabels[i]]*rows), test_size=0.3, random_state=42)

input=[]
target=[]

for i in range(7):
    for j in dfs[headlabels[i]]:
        input.append(j)
        target.append(i)

input=np.array(input)
target=np.array(target)

input_train=[]
target_train=[]

for i in range(7):
    for j in X_train[i]:
        input_train.append(j)
        target_train.append(i)

input_train=np.array(input_train)
target_train=np.array(target_train)

input_val=[]
target_val=[]

for i in range(7):
    for j in X_test[i]:
        input_val.append(j)
        target_val.append(i)

input_val=np.array(input_val)
target_val=np.array(target_val)
    


# In[ ]:





# In[15]:


original_input_train, original_input_test, original_output_train, original_output_test = train_test_split(input, target, test_size=0.2, random_state=42)


# In[16]:


from random import seed
from random import randrange
 
# Split a dataset into 2 folds
def cross_validation_split(input, target, folds):
	input_split=[]
	input_copy=list(input.copy())
	target_split=[]
	target_copy=list(target.copy())
	fold_size=int(len(input)/folds)
	for i in range(folds):
		fold_input=[]
		fold_target=[]
		while len(fold_input) < fold_size:
			index = randrange(len(input_copy))
			fold_input.append(input_copy.pop(index))
			fold_target.append(target_copy.pop(index))
		input_split.append(fold_input)
		target_split.append(fold_target)
	return input_split, target_split

seed(1)
folds_input, folds_target = cross_validation_split(original_input_train, original_output_train, 2)


# In[ ]:


import tensorflow_addons as tfa
import numpy as np

# Define the custom loss function
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def call(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# Add projection head to the encoder
def add_projection_head(encoder, projection_units=128):
    inputs = keras.Input(shape=(32, 32, 3))
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head")
    return model

# Create classifier
def create_classifier(encoder, trainable=True, hidden_units=512, dropout_rate=0.5, learning_rate=0.001):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=(32, 32, 3))
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(7, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# Assuming folds_input and folds_target are defined and contain your data
# For this example, we'll use dummy data
# folds_input = [np.random.rand(100, 32, 32, 3) for _ in range(3)]
# folds_target = [np.random.randint(0, 7, 100) for _ in range(3)]

accuracies = []
learning_rate = 0.001
batch_size = 800
hidden_units = 512
projection_units = 128
num_epochs = 100
dropout_rate = 0.5
temperature = 0.05

for i in range(2):
    input_train = []
    input_val = []
    target_train = []
    target_val = []

    for sample in range(len(folds_input[i])):
        input_val.append(folds_input[i][sample])
        target_val.append(folds_target[i][sample])

    for j in range(2):
        if j == i:
            continue
        for sample in range(len(folds_input[j])):
            input_train.append(folds_input[j][sample])
            target_train.append(folds_target[j][sample])

    input_train = np.array(input_train)
    input_val = np.array(input_val)
    target_train = np.array(target_train)
    target_val = np.array(target_val)


    encoder = create_encoder()
    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupervisedContrastiveLoss(temperature),
        run_eagerly=True
    )

    encoder_with_projection_head.summary()

    history = encoder_with_projection_head.fit(
        x=input_train, y=target_train, batch_size=batch_size, epochs=num_epochs
    )

    classifier = create_classifier(encoder, trainable=False)
    history = classifier.fit(x=input_train, y=target_train, batch_size=batch_size, epochs=num_epochs)

    loss, accuracy = classifier.evaluate(input_val, target_val)
    accuracies.append(round(accuracy * 100, 2))

print("Accuracies:", accuracies)


# In[ ]:


target_pred=classifier.predict(original_input_test)
target_predict = np.argmax(target_pred, axis=1)
print("Accuracy: "+str(accuracy_score(target_predict,original_output_test)))
print("F1 Score: "+str(f1_score(target_predict,original_output_test,average='macro')))


# 10-fold cross validation

# In[ ]:


from random import seed
from random import randrange
 
# Split a dataset into 2 folds
def cross_validation_split(input, target, folds):
	input_split=[]
	input_copy=list(input.copy())
	target_split=[]
	target_copy=list(target.copy())
	fold_size=int(len(input)/folds)
	for i in range(folds):
		fold_input=[]
		fold_target=[]
		while len(fold_input) < fold_size:
			index = randrange(len(input_copy))
			fold_input.append(input_copy.pop(index))
			fold_target.append(target_copy.pop(index))
		input_split.append(fold_input)
		target_split.append(fold_target)
	return input_split, target_split

seed(1)
folds_input, folds_target = cross_validation_split(original_input_train, original_output_train, 10)


# In[ ]:


import tensorflow_addons as tfa
import numpy as np

# Define the custom loss function
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def call(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# Add projection head to the encoder
def add_projection_head(encoder, projection_units=128):
    inputs = keras.Input(shape=(32, 32, 3))
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head")
    return model

# Create classifier
def create_classifier(encoder, trainable=True, hidden_units=512, dropout_rate=0.5, learning_rate=0.001):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=(32, 32, 3))
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(7, activation=import tensorflow_addons as tfa
import numpy as np

# Define the custom loss function
class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def call(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# Add projection head to the encoder
def add_projection_head(encoder, projection_units=128):
    inputs = keras.Input(shape=(32, 32, 3))
    features = encoder(inputs)
    outputs = layers.Dense(projection_units, activation="relu")(features)
    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head")
    return model

# Create classifier
def create_classifier(encoder, trainable=True, hidden_units=512, dropout_rate=0.5, learning_rate=0.001):
    for layer in encoder.layers:
        layer.trainable = trainable

    inputs = keras.Input(shape=(32, 32, 3))
    features = encoder(inputs)
    features = layers.Dropout(dropout_rate)(features)
    features = layers.Dense(hidden_units, activation="relu")(features)
    features = layers.Dropout(dropout_rate)(features)
    outputs = layers.Dense(7, activation="softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# Assuming folds_input and folds_target are defined and contain your data
# For this example, we'll use dummy data
# folds_input = [np.random.rand(100, 32, 32, 3) for _ in range(3)]
# folds_target = [np.random.randint(0, 7, 100) for _ in range(3)]

accuracies = []
learning_rate = 0.001
batch_size = 800
hidden_units = 512
projection_units = 128
num_epochs = 100
dropout_rate = 0.5
temperature = 0.05

for i in range(2):
    input_train = []
    input_val = []
    target_train = []
    target_val = []

    for sample in range(len(folds_input[i])):
        input_val.append(folds_input[i][sample])
        target_val.append(folds_target[i][sample])

    for j in range(2):
        if j == i:
            continue
        for sample in range(len(folds_input[j])):
            input_train.append(folds_input[j][sample])
            target_train.append(folds_target[j][sample])

    input_train = np.array(input_train)
    input_val = np.array(input_val)
    target_train = np.array(target_train)
    target_val = np.array(target_val)


    encoder = create_encoder()
    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupervisedContrastiveLoss(temperature),
        run_eagerly=True
    )

    encoder_with_projection_head.summary()

    history = encoder_with_projection_head.fit(
        x=input_train, y=target_train, batch_size=batch_size, epochs=num_epochs
    )

    classifier = create_classifier(encoder, trainable=False)
    history = classifier.fit(x=input_train, y=target_train, batch_size=batch_size, epochs=num_epochs)

    loss, accuracy = classifier.evaluate(input_val, target_val)
    accuracies.append(round(accuracy * 100, 2))

print("Accuracies:", accuracies)
"softmax")(features)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10-classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model

# Assuming folds_input and folds_target are defined and contain your data
# For this example, we'll use dummy data
# folds_input = [np.random.rand(100, 32, 32, 3) for _ in range(3)]
# folds_target = [np.random.randint(0, 7, 100) for _ in range(3)]

accuracies = []
learning_rate = 0.001
batch_size = 800
hidden_units = 512
projection_units = 128
num_epochs = 20
dropout_rate = 0.5
temperature = 0.05

for i in range(10):
    input_train = []
    input_val = []
    target_train = []
    target_val = []

    for sample in range(len(folds_input[i])):
        input_val.append(folds_input[i][sample])
        target_val.append(folds_target[i][sample])

    for j in range(10):
        if j == i:
            continue
        for sample in range(len(folds_input[j])):
            input_train.append(folds_input[j][sample])
            target_train.append(folds_target[j][sample])

    input_train = np.array(input_train)
    input_val = np.array(input_val)
    target_train = np.array(target_train)
    target_val = np.array(target_val)


    encoder = create_encoder()
    encoder_with_projection_head = add_projection_head(encoder)
    encoder_with_projection_head.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=SupervisedContrastiveLoss(temperature),
        run_eagerly=True
    )

    encoder_with_projection_head.summary()

    history = encoder_with_projection_head.fit(
        x=input_train, y=target_train, batch_size=batch_size, epochs=num_epochs
    )

    classifier = create_classifier(encoder, trainable=False)
    history = classifier.fit(x=input_train, y=target_train, batch_size=batch_size, epochs=num_epochs)

    loss, accuracy = classifier.evaluate(input_val, target_val)
    accuracies.append(round(accuracy * 100, 2))

print("Accuracies:", accuracies)


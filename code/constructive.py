#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import re
from math import sqrt, atan2
import pywt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import scipy.signal as signal
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


# In[18]:


#loading the dataset
import sys
folder =sys.argv[1] # change the folder as per requirement
feat_train_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_train.csv'))
label_train_amp=pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_train.csv'))
feat_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/X_test.csv'))
label_test_amp = pd.read_csv(os.path.abspath(f'./training_testing_data/{folder}/y_test.csv'))


# In[19]:


# encode label 
label_train_amp.headlabel.replace(('Forward', 'Looking Up', 'Nodding', 'Looking Down', 'Shaking','Looking Left', 'Looking Right' ), (1, 2,3, 4,5,6,7), inplace=True)
label_test_amp.headlabel.replace(('Forward', 'Looking Up', 'Nodding', 'Looking Down', 'Shaking','Looking Left', 'Looking Right' ), (1, 2,3, 4,5,6,7), inplace=True)


# In[24]:


# reduce feature dimension 
selector = SelectKBest(f_classif, k=100)
feat_train_amp_100 = selector.fit_transform(feat_train_amp, label_train_amp)
feat_test_amp_100 = selector.fit_transform(feat_test_amp, label_test_amp)


# In[26]:


# split into train test 
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(feat_train_amp_100, label_train_amp, test_size=0.30, random_state=42)
train_data, val_data, train_label, val_label= train_test_split(train_data, train_label, test_size=0.10, random_state=42)


# In[27]:


class CasperNetwork(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(CasperNetwork, self).__init__()
        
        # create input layer with a randomly initialized weight
        self.Initial = torch.nn.Linear(n_input, n_output) 
        self.Initial.weight.data = self.initialize_weights(n_input, n_output, 
                                                    -0.7, 0.7)
        
        # This list contains all the input connections to hidden neurons
        self.old_input_neurons = nn.ModuleList([]) 
        # This list contains all the ouput connections from previous neurons
        self.old_output_neurons = nn.ModuleList([]) 
        self.n_neurons = 0
        
        self.input_size = n_input
        self.output_size = n_output

        # initialize casper network with no hidden neurons
        self.L1 = None
        self.L2 = None

        self.output_layer = torch.nn.Linear(n_input, n_output) 
        self.output_layer.weight.data = self.initialize_weights(n_input, 
                                                                n_output, 
                                                                -0.7, 0.7)
    def forward(self, x):
        # calculate output from input layer
        out = x

        # if there are no hidden nerons, simply return the output
        if len(self.old_input_neurons) == 0:
            if self.L1 == None:
                
                # if no neurons has been inserted, simply pass input to ouput
                return self.Initial(x)
            
            # if there is a single hidden neuron, 
            # then add its output to the output layer
            else:
                
                temp = torch.tanh(self.L1(x))
                temp = torch.tanh(self.L2(temp))
                out = torch.cat((out, temp), 1)
                
        
        else:
            # if there is more than 1 hidden neuron, loop through the list of
            # casper neurons and add their output to the final output layer
            for index in range(0, len(self.old_input_neurons)):
                
                # calculate the inputs to the single casper neuron
                previous = self.old_input_neurons[index](x)
                
                # concatenate the output from the single casper neuron to the 
                # outputs of all previous neurons
                out = torch.cat((out, self.old_output_neurons[index]
                                (torch.tanh(previous))), 1) 

                # concatenate the single neuron to the input 
                x = torch.cat((x, previous), 1)
                
            # calculate the output from the most recent casper neuron 
            # add them to the final output layer
            new_neuron_input = torch.tanh(self.L1(x))
            new_neuron_output = torch.tanh(self.L2(new_neuron_input))
            out = torch.cat((out, new_neuron_output), 1)
          
        return self.output_layer(out)
    
     
    # adds new casper neuron to the network, which would be 2 linear layers
    # layer1: (input, 1) layer2: (1, output), 
    # the flow would be inputs -> 1 neuron -> output
    def add_layer(self):
        self.n_neurons += 1
        
        # concatenate all outputs from the hidden neurons and original input
        # to go into the output neurons
        previous_weights = self.output_layer.weight.data
        total_outputs = self.n_neurons + self.input_size
        self.output_layer = torch.nn.Linear(total_outputs , self.output_size) 
        
        # copy over the previsouly learnt weights, and initialize random
        # weight values for new neurons
        self.output_layer.weight.data = self.copy_initialize_weights(
                                                            previous_weights, 
                                                            total_outputs, 
                                                            self.output_size, 
                                                            -0.1, 0.1)
        
        # create the layers for the new casper neuron
        new_layer_in = torch.nn.Linear(self.input_size + self.n_neurons - 1, 1)
        # we pass it through another neuron in order to create an per neuron
        # learning rate for the final layer
        new_layer_out = torch.nn.Linear(1, 1)
        
        total_inputs = self.input_size + self.n_neurons - 1
        
        # initialize weights
        new_layer_in.weight.data = self.initialize_weights(total_inputs, 
                                                         1, -0.1, 0.1)
        new_layer_out.weight.data = self.initialize_weights(1, 1, -0.1, 0.1)
        
        # assign the layers to the network
        if self.L1 == None and self.L2 == None:
            self.L1 = new_layer_in
            self.L2 = new_layer_out
        
        else:
            self.old_input_neurons.append(self.L1)
            self.old_output_neurons.append(self.L2)
            self.L1 = new_layer_in
            self.L2 = new_layer_out
         
        
        
    # create a list of weights for initialization
    def initialize_weights(self, n_input, n_output, lower, upper):
        final = []
        for inputs in range(0, n_output):
            weights = []
            for value in range(0, n_input):
                weights.append(np.random.uniform(lower, upper))
            final.append(weights)
        return torch.Tensor(final)
    
    # Creates a list of weights for initialization, but also copies over the 
    # previous weights avoiding the need to relearn
    def copy_initialize_weights(self, previous_weight, n_input, n_output, 
                                lower, upper):
        final = []
        
        for row in range(0, n_output):
            weights = []
            
            for value in range(0, len(previous_weight[row])):
                weights.append(previous_weight[row][value])
                    
            for new_weight in range(0, n_input - len(previous_weight[row])):
                weights.append(np.random.uniform(lower, upper))

            final.append(weights)
        return torch.Tensor(final)
    
    
    def applyWeightDecay(self, decay):
        self.Initial.weight.data *= decay
        
        if self.L1 != None:
            self.L1.weight.data *= decay
            self.L2.weight.data *= decay
            
        if len(self.old_input_neurons) != 0:
            for layers in self.old_input_neurons:
                layers.weight.data *= decay
                
            for layers in self.old_output_neurons:
                layers.weight.data *= decay 


# In[30]:


X = Variable(torch.Tensor(train_data).float())
Y = Variable(torch.Tensor(train_label.values).long())
VX = Variable(torch.Tensor(val_data).float())
VY = Variable(torch.Tensor(val_label.values).long())
Y=Y.flatten()-1
VY=VY.flatten()-1


# In[32]:


num_epochs = 1360
output_neurons = 7
input_neurons =100
weight_decay = 0.998
threshold=0.5

L1 = 0.005
L2 = 0.001
L3 = 0.0005


# In[33]:


net = CasperNetwork(input_neurons, output_neurons)
# net=net.to(device)


# define loss function
loss_func = torch.nn.CrossEntropyLoss()



# define optimiser with per layer learning rates
# optimiser without any hidden neurons
optimiser = optim.Rprop([
                {'params': net.Initial.parameters(), 'lr' : L1},
                {'params': net.output_layer.parameters()},
                {'params': net.old_input_neurons.parameters()},
                {'params': net.old_output_neurons.parameters()}], 
                lr = L3, etas = (0.5, 1.2), step_sizes=(1e-06, 50))


# In[35]:


batch_size=50


# In[ ]:


all_losses = []
val_loss=[]


previous_loss = None

# train a neural network
for epoch in range(num_epochs):
    permutation = torch.randperm(X.size()[0])
    temp_loss=0
    idx=0
#     torch.cuda.empty_cache()
    for i in range(0,X.size()[0], batch_size):
        idx=idx+1
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X[indices], Y[indices]
#         batch_x=batch_x.to(device)
#         batch_y=batch_y.to(device)
#         net.to(device)
    # Perform forward pass: compute predicted y by passing x to the model.
        Y_pred = net(batch_x)
    #     Y_pred=(Y_pred)
    #     print(Y_pred)
    #     print(Y_pred.shape)
    #     print(Y.shape)

        # Compute loss
        loss = loss_func(Y_pred, batch_y)

        temp_loss=temp_loss+loss
        
    # Clear the gradients before running the backward pass.
    net.zero_grad()
    
    ll=(temp_loss)/idx

    # Perform backward pass
    ll.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()
    net.applyWeightDecay(weight_decay)
    
    val_temp=0
    
    idxv=0
    permutation_v = torch.randperm(VX.size()[0])
    for i in range(0,VX.size()[0], batch_size):
        idxv=idxv+1
        net.eval()
        indices_v = permutation_v[i:i+batch_size]
        batch_x_val, batch_y_val = VX[indices_v], VY[indices_v]
#         batch_x_val=batch_x_val.to(device)
#         batch_y_val=batch_y_val.to(device)
#         net.to(device)
    # Perform forward pass: compute predicted y by passing x to the model.
        Y_pred_v = net(batch_x_val)

        # Compute loss
        loss = loss_func(Y_pred_v, batch_y_val)

        val_temp=val_temp+loss
    
    all_losses.append((temp_loss)/idx)
    val_loss.append((val_temp)/idxv)
    # print progress
    if epoch % 40 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(Y_pred, 1)
        _, predicted_v = torch.max(Y_pred_v, 1)
#         predicted=predicted.to(device)
#             print(predicted)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.cpu().data.numpy() == batch_y.cpu().data.numpy()
        
        total_v = predicted_v.size(0)
        correct_v = predicted_v.cpu().data.numpy() == batch_y_val.cpu().data.numpy()
        
#         print(correct)
        print('Epoch [%d/%d] Train Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, all_losses[-1], 100 * sum(correct)/total))
        print('Epoch [%d/%d] Val Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, val_loss[-1], 100 * sum(correct_v)/total_v))
    

        # if the rate to which the loss value decreases slows beyond a certain
        # threshold, then add a casper neuron
        if (previous_loss != None and previous_loss > all_losses[-1] and 
                                    previous_loss - all_losses[-1] < threshold) :

            net.add_layer()

            # adding custom learning rates to hidden neurons
            optimiser = optim.Rprop([
                {'params': net.Initial.parameters()},
                {'params': net.old_input_neurons.parameters()},
                {'params': net.old_output_neurons.parameters()},
                {'params': net.output_layer.parameters()},
                {'params': net.L1.parameters(), 'lr': L1},
                {'params': net.L2.parameters(), 'lr': L2}], 
                lr = L3, etas = (0.5, 1.2), step_sizes=(1e-06, 50))

        previous_loss = all_losses[-1]



# In[41]:


from sklearn.metrics import f1_score
# create Tensors to hold inputs and outputs, and wrap them in Variables,
# as Torch only trains neural network on Variables
X_test = Variable(torch.Tensor(test_data).float())
Y_test = Variable(torch.Tensor(test_label.values).long())
# X_test=X_test.to(device)
# Y_test=Y_test.to(device)
Y_test=Y_test.flatten()-1

# test the neural network using testing data
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.cpu().data.numpy() == Y_test.cpu().data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
# Calculate F1-score
f1 = f1_score(Y_test, predicted_test, average='weighted')
print('F1-score:', f1)


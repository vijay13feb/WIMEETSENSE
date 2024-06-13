#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd 
import numpy as np
import os
import pickle


# Training and Testing for S1 setup

# In[10]:


main_dir = os.path.abspath('./dopplers')
save = os.path.abspath('./S1')
main_list= os.listdir(main_dir)
main_list.sort()
forward=[]
looking_down=[]
looking_up=[]
looking_left=[]
looking_right=[]
nodding=[]
shaking=[]
for i in main_list:
    sub_dir= f'{main_dir}/{i}'
    sub_list= os.listdir(sub_dir)
    sub_list.sort()
    for j in sub_list:
        file_dir= f'{main_dir}/{i}/{j}'
        file_list=os.listdir(file_dir)
        for li in file_list:
            path = f'{main_dir}/{i}/{j}/{li}'
            with open(path, 'rb') as fp:
                H_est = pickle.load(fp)
            if 'Forward' in li:
                forward.append(H_est)
            elif 'Looking Down' in li:
                looking_down.append(H_est)
            elif 'Looking Up' in li:
                looking_up.append(H_est)
            elif 'Looking Left' in li:
                looking_left.append(H_est)
            elif 'Looking Right' in li:
                looking_right.append(H_est)
            elif 'Nodding' in li:
                nodding.append(H_est)
            elif 'Shaking' in li:
                shaking.append(H_est)
fr = np.vstack(forward)
path = f'{save}/Forward.txt'
with open(path, 'wb') as file:
    pickle.dump(fr, file)

ld = np.vstack(looking_down)
path = f'{save}/Looking Down.txt'
with open (path, 'wb') as file:
    pickle.dump(ld, file)

lu = np.vstack(looking_up)
path = f'{save}/Looking Up.txt'
with open (path, 'wb') as file:
    pickle.dump(lu, file)
    
ll = np.vstack(looking_left)
path = f'{save}/Looking Left.txt'
with open (path, 'wb') as file:
    pickle.dump(ll, file)
    
lr = np.vstack(looking_right)
path = f'{save}/Looking Right.txt'
with open (path, 'wb') as file:
    pickle.dump(lr, file)

no = np.vstack(nodding)
path = f'{save}/Nodding.txt'
with open (path, 'wb') as file:
    pickle.dump(no, file)
    
sh = np.vstack(shaking)
path = f'{save}/Shaking.txt'
with open(path, 'wb') as file:
    pickle.dump(sh, file)
            


# In[ ]:


main_dir = os.path.abspath('./dopplers')
save = os.path.abspath('./S2')
main_list= os.listdir(main_dir)
main_list.sort()
forward=[]
looking_down=[]
looking_up=[]
looking_left=[]
looking_right=[]
nodding=[]
shaking=[]
for i in main_list:
    sub_dir= f'{main_dir}/{i}'
    sub_list= os.listdir(sub_dir)
    sub_list.sort()
    for j in sub_list:
        file_dir= f'{main_dir}/{i}/{j}'
        file_list=os.listdir(file_dir)
        for li in file_list:
            path = f'{main_dir}/{i}/{j}/{li}'
            with open(path, 'rb') as fp:
                H_est = pickle.load(fp)
            if 'Forward' in li:
                forward.append(H_est)
            elif 'Looking Down' in li:
                looking_down.append(H_est)
            elif 'Looking Up' in li:
                looking_up.append(H_est)
            elif 'Looking Left' in li:
                looking_left.append(H_est)
            elif 'Looking Right' in li:
                looking_right.append(H_est)
            elif 'Nodding' in li:
                nodding.append(H_est)
            elif 'Shaking' in li:
                shaking.append(H_est)
fr = np.vstack(forward)
path = f'{save}/Forward.txt'
with open(path, 'wb') as file:
    pickle.dump(fr, file)

ld = np.vstack(looking_down)
path = f'{save}/Looking Down.txt'
with open (path, 'wb') as file:
    pickle.dump(ld, file)

lu = np.vstack(looking_up)
path = f'{save}/Looking Up.txt'
with open (path, 'wb') as file:
    pickle.dump(lu, file)
    
ll = np.vstack(looking_left)
path = f'{save}/Looking Left.txt'
with open (path, 'wb') as file:
    pickle.dump(ll, file)
    
lr = np.vstack(looking_right)
path = f'{save}/Looking Right.txt'
with open (path, 'wb') as file:
    pickle.dump(lr, file)

no = np.vstack(nodding)
path = f'{save}/Nodding.txt'
with open (path, 'wb') as file:
    pickle.dump(no, file)
    
sh = np.vstack(shaking)
path = f'{save}/Shaking.txt'
with open(path, 'wb') as file:
    pickle.dump(sh, file)


# In[ ]:


main_dir = os.path.abspath('./dopplers')
save = os.path.abspath('./S3')
main_list= os.listdir(main_dir)
main_list.sort()
forward=[]
looking_down=[]
looking_up=[]
looking_left=[]
looking_right=[]
nodding=[]
shaking=[]
for i in main_list:
    sub_dir= f'{main_dir}/{i}'
    sub_list= os.listdir(sub_dir)
    sub_list.sort()
    for j in sub_list:
        file_dir= f'{main_dir}/{i}/{j}'
        file_list=os.listdir(file_dir)
        for li in file_list:
            path = f'{main_dir}/{i}/{j}/{li}'
            with open(path, 'rb') as fp:
                H_est = pickle.load(fp)
            if 'Forward' in li:
                forward.append(H_est)
            elif 'Looking Down' in li:
                looking_down.append(H_est)
            elif 'Looking Up' in li:
                looking_up.append(H_est)
            elif 'Looking Left' in li:
                looking_left.append(H_est)
            elif 'Looking Right' in li:
                looking_right.append(H_est)
            elif 'Nodding' in li:
                nodding.append(H_est)
            elif 'Shaking' in li:
                shaking.append(H_est)
fr = np.vstack(forward)
path = f'{save}/Forward.txt'
with open(path, 'wb') as file:
    pickle.dump(fr, file)

ld = np.vstack(looking_down)
path = f'{save}/Looking Down.txt'
with open (path, 'wb') as file:
    pickle.dump(ld, file)

lu = np.vstack(looking_up)
path = f'{save}/Looking Up.txt'
with open (path, 'wb') as file:
    pickle.dump(lu, file)
    
ll = np.vstack(looking_left)
path = f'{save}/Looking Left.txt'
with open (path, 'wb') as file:
    pickle.dump(ll, file)
    
lr = np.vstack(looking_right)
path = f'{save}/Looking Right.txt'
with open (path, 'wb') as file:
    pickle.dump(lr, file)

no = np.vstack(nodding)
path = f'{save}/Nodding.txt'
with open (path, 'wb') as file:
    pickle.dump(no, file)
    
sh = np.vstack(shaking)
path = f'{save}/Shaking.txt'
with open(path, 'wb') as file:
    pickle.dump(sh, file)


# In[7]:


# S4 Setup 
# combine semi data
main_dir = os.path.abspath('./dopplers/semi')
save = os.path.abspath('./S4')
main_list= os.listdir(main_dir)
main_list.sort()
forward=[]
looking_down=[]
looking_up=[]
looking_left=[]
looking_right=[]
nodding=[]
shaking=[]

sub_list= os.listdir(main_dir)
sub_list.sort()
for j in sub_list:
    file_dir= f'{main_dir}/{j}'
    file_list=os.listdir(file_dir)
    for li in file_list:
        path = f'{main_dir}/{j}/{li}'
        with open(path, 'rb') as fp:
            H_est = pickle.load(fp)
        if 'Forward' in li:
            forward.append(H_est)
        elif 'Looking Down' in li:
            looking_down.append(H_est)
        elif 'Looking Up' in li:
            looking_up.append(H_est)
        elif 'Looking Left' in li:
            looking_left.append(H_est)
        elif 'Looking Right' in li:
            looking_right.append(H_est)
        elif 'Nodding' in li:
            nodding.append(H_est)
        elif 'Shaking' in li:
            shaking.append(H_est)
fr_semi = np.vstack(forward)
# path = f'{save}/Forward.txt'
# with open(path, 'wb') as file:
# pickle.dump(fr, path)

ld_semi = np.vstack(looking_down)
# path = f'{save}/Looking Down.txt'
# with open (path, 'wb') as file:
# pickle.dump(ld, path)

lu_semi = np.vstack(looking_up)
# path = f'{save}/Looking Up.txt'
# with open (path, 'wb') as file:
# pickle.dump(lu, path)

ll_semi = np.vstack(looking_left)
# path = f'{save}/Looking Left.txt'
# with open (path, 'wb') as file:
# pickle.dump(ll, path)

lr_semi = np.vstack(looking_right)
# path = f'{save}/Looking Right.txt'
# with open (path, 'wb') as file:
# pickle.dump(lr, path)

no_semi = np.vstack(nodding)
# path = f'{save}/Nodding.txt'
# with open (path, 'wb') as file:
# pickle.dump(no, path)

sh_semi = np.vstack(shaking)
# path = f'{save}/Shaking.txt'
# with open(path, 'wb') as file:
# pickle.dump(sh, path)
# combine wild data
main_dir = os.path.abspath('./dopplers/wild')
save = os.path.abspath('./S4')
main_list= os.listdir(main_dir)
main_list.sort()
forward=[]
looking_down=[]
looking_up=[]
looking_left=[]
looking_right=[]
nodding=[]
shaking=[]

sub_list= os.listdir(main_dir)
sub_list.sort()
for j in sub_list:
    file_dir= f'{main_dir}/{j}'
    file_list=os.listdir(file_dir)
    for li in file_list:
        path = f'{main_dir}/{j}/{li}'
        with open(path, 'rb') as fp:
            H_est = pickle.load(fp)
        if 'Forward' in li:
            forward.append(H_est)
        elif 'Looking Down' in li:
            looking_down.append(H_est)
        elif 'Looking Up' in li:
            looking_up.append(H_est)
        elif 'Looking Left' in li:
            looking_left.append(H_est)
        elif 'Looking Right' in li:
            looking_right.append(H_est)
        elif 'Nodding' in li:
            nodding.append(H_est)
        elif 'Shaking' in li:
            shaking.append(H_est)
fr_wild = np.vstack(forward)
# path = f'{save}/Forward.txt'
# with open(path, 'wb') as file:
# pickle.dump(fr, path)

ld_wild = np.vstack(looking_down)
# path = f'{save}/Looking Down.txt'
# with open (path, 'wb') as file:
# pickle.dump(ld, path)

lu_wild = np.vstack(looking_up)
# path = f'{save}/Looking Up.txt'
# with open (path, 'wb') as file:
# pickle.dump(lu, path)

ll_wild = np.vstack(looking_left)
# path = f'{save}/Looking Left.txt'
# with open (path, 'wb') as file:
# pickle.dump(ll, path)

lr_wild = np.vstack(looking_right)
# path = f'{save}/Looking Right.txt'
# with open (path, 'wb') as file:
# pickle.dump(lr, path)

no_wild = np.vstack(nodding)
# path = f'{save}/Nodding.txt'
# with open (path, 'wb') as file:
# pickle.dump(no, path)

sh_wild = np.vstack(shaking)
# path = f'{save}/Shaking.txt'
# with open(path, 'wb') as file:
# pickle.dump(sh, path)
fr= np.vstack((fr_semi, fr_wild))
ld= np.vstack((ld_semi, ld_wild))
lu= np.vstack((lu_semi, lu_wild))
ll= np.vstack((ll_semi, ll_wild))
lr= np.vstack((lr_semi, lr_wild))
no= np.vstack((no_semi, no_wild))
sh= np.vstack((sh_semi, sh_wild))


path = f'{save}/Forward.txt'
with open(path, 'wb') as file:
    pickle.dump(fr, file)


path = f'{save}/Looking Down.txt'
with open (path, 'wb') as file:
    pickle.dump(ld, file)


path = f'{save}/Looking Up.txt'
with open (path, 'wb') as file:
    pickle.dump(lu, file)


path = f'{save}/Looking Left.txt'
with open (path, 'wb') as file:
    pickle.dump(ll, file)

path = f'{save}/Looking Right.txt'
with open (path, 'wb') as file:
    pickle.dump(lr, file)


path = f'{save}/Nodding.txt'
with open (path, 'wb') as file:
    pickle.dump(no, file)


path = f'{save}/Shaking.txt'
with open(path, 'wb') as file:
    pickle.dump(sh, file)


# In[ ]:





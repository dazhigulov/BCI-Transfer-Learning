# ==============================================================================
#  created by @berdakh.abibullaev
#        and 
#  developed for deep transfer learning by @qasymjomart
# ==============================================================================
import itertools
import numpy as np
import time, copy, pdb
import pandas as pd 
import mne 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.utils import shuffle 

#%% sklearn standard scaler  
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        # Save the original shape to reshape the flattened X later
        # back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to it's original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X 
#%%

def subject_specific(subjectIndex, d1, dataset):
    # returns torch tensors with extract positive and negative classes  
    """Leave one subject out wrapper        
    Input: d1 - is list consisting of subject-specific epochs in MNE structure
    Example usage:
        subjectIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]            
        for sub in subjectIndex:
            xvalid = subject_specific(sub, d1)          
    """
    if dataset == 'NU':
        pos_str = 'Target'
        neg_str = 'NonTarget'
    elif dataset == 'EPFL':
        pos_str = 'Target'
        neg_str = 'NonTarget'
    elif dataset == 'ALS':
        pos_str = 'Target'
        neg_str = 'NonTarget'
    elif dataset == 'BNCI':        
        pos_str = 'Target'
        neg_str = 'NonTarget'
    
    data, pos, neg = [], [] , []    
    if len(subjectIndex) > 1: # multiple subjects             
        for jj in subjectIndex:                
            print('Loading subjects:', jj)   
            dat = d1[jj]                                     
            pos.append(dat[pos_str].get_data())
            neg.append(dat[neg_str].get_data())  
    else: 
        print('Loading subject:', subjectIndex[0])  
        dat = d1[subjectIndex[0]]
        pos.append(dat[pos_str].get_data())
        neg.append(dat[neg_str].get_data())  

    for ii in range(len(pos)):
        # subject specific upsampling of minority class 
        targets = pos[ii]              
        for j in range(neg[ii].shape[0]//pos[ii].shape[0]): 
            targets = np.concatenate([pos[ii], targets])                    
        pos[ii] = targets  
        
        X = np.concatenate([pos[ii].astype('float32'), neg[ii].astype('float32')])            
        Y = np.concatenate([np.ones(pos[ii].shape[0]).astype('float32'), 
                            np.zeros(neg[ii].shape[0]).astype('float32')])       
        data.append(dict(xtrain = X, ytrain = Y))            
    return data

# ### Leave one-subject out cross-validation 
# In[6]:
def leave1out(sub, subjectIndex, d1, dataset):
    """
    Inputs:xz
        sub = subject index for leaving out
        subjectIndex = all subject index 
        d1 = dataset [list of MNE data structure]        
    Returns:
        xvalid = leave one subject out dataset
        xtrain = training set  
        
    Example usage:
    ------------------------------
    import pickle
    filename = 'data_allsubjects'
    with open(filename, 'rb') as handle:
    d1 = pickle.load(handle)  
    
    subjectIndex = [0,1,2,3,4,5,6,7,8,9,10] 
    for sub in subjectIndex:
        data = leave1out(sub, subjectIndex, d1)
        out = get_data(data, batch_size = 64, image = True, lstm = False, raw = False)   
    """   
    subject = copy.deepcopy(subjectIndex)        
    # load the validation data 
    xval   = subject_specific([sub], d1, dataset)       
    xvalid, yvalid = map(torch.FloatTensor, (xval[0]['xtrain'], xval[0]['ytrain']))

    # leave one subject index out 
    subject.remove(sub)
    print(subject)        
    # load subjects (N-1)
    data = subject_specific(subject, d1, dataset)      
    # concatenate (N-1) data and return xtrain/valid
    for jj in range(len(data)):             
        if jj == 0:
            xtrain = data[jj]['xtrain']
            ytrain = data[jj]['ytrain'] 
            
            # normalize 
            scaler = NDStandardScaler()
            xtrain = scaler.fit_transform(xtrain)  
            xtrain, ytrain = map(torch.FloatTensor, (xtrain, ytrain))      
            
        else:
            xtrain = np.concatenate([xtrain, data[jj]['xtrain']])            
            ytrain = np.concatenate([ytrain, data[jj]['ytrain']])        
            
            # normalize 
            scaler = NDStandardScaler()
            xtrain = scaler.fit_transform(xtrain)  
            xtrain, ytrain = map(torch.FloatTensor, (xtrain, ytrain))
            
    return xtrain, ytrain, xvalid, yvalid

def importseveralsubjects(subjectIndex, d1, dataset):
    """
    Inputs:xz
        sub = subject index for leaving out
        subjectIndex = all subject index 
        d1 = dataset [list of MNE data structure]        
    Returns:
        xvalid = leave one subject out dataset
        xtrain = training set  
        
    Example usage:
    ------------------------------
    import pickle
    filename = 'data_allsubjects'
    with open(filename, 'rb') as handle:
    d1 = pickle.load(handle)  
    
    subjectIndex = [0,1,2,3,4,5,6,7,8,9,10] 
    for sub in subjectIndex:
        data = leave1out(sub, subjectIndex, d1)
        out = get_data(data, batch_size = 64, image = True, lstm = False, raw = False)   
    """   
    subject = copy.deepcopy(subjectIndex)        
    # load the validation data 
    
    print(subject)        
    # load subjects (N-1)
    data = subject_specific(subject, d1, dataset)      
    # concatenate (N-1) data and return xtrain/valid
    for jj in range(len(data)):             
        if jj == 0:
            xtrain = data[jj]['xtrain']
            ytrain = data[jj]['ytrain'] 
            
            # normalize 
            scaler = NDStandardScaler()
            xtrain = scaler.fit_transform(xtrain)
#             x_train = xtrain[0]['xtrain']
#             y_train = xtrain[0]['ytrain']     
            
        else:
            xtrain = np.concatenate([xtrain, data[jj]['xtrain']])            
            ytrain = np.concatenate([ytrain, data[jj]['ytrain']])        
            
            # normalize 
            scaler = NDStandardScaler()
            xtrain = scaler.fit_transform(xtrain)  
#             x_train = xtrain[0]['xtrain']
#             y_train = xtrain[0]['ytrain']
            
    return xtrain, ytrain

def pad_with_zeros(x, desired_height=224, desired_width=224):
    x_size = x.shape
    delta_w = desired_width - x_size[1]
    delta_h = desired_height - x_size[0]
    if (delta_h % 2 == 0) and (delta_w % 2 == 0):
        new_x = np.pad(x, ((delta_h//2, delta_h//2), (delta_w//2, delta_w//2)), 'constant', constant_values=(0, 0))
    elif (delta_h % 2 == 1) and (delta_w % 2 == 0):
        new_x = np.pad(x, ((delta_h//2, delta_h//2 + 1), (delta_w//2, delta_w//2)), 'constant', constant_values=(0, 0))
    elif (delta_h % 2 == 0) and (delta_w % 2 == 1):
        new_x = np.pad(x, ((delta_h//2, delta_h//2), (delta_w//2, delta_w//2 +1)), 'constant', constant_values=(0, 0))
    else:
        new_x = np.pad(x, ((delta_h//2, delta_h//2 + 1), (delta_w//2, delta_w//2 +1)), 'constant', constant_values=(0, 0))
    return new_x

def pad_with_zeros_below(x, desired_height=224, desired_width=224):
    x_size = x.shape
    delta_w = desired_width - x_size[1]
    delta_h = desired_height - x_size[0]
    new_x = np.pad(x, ((0,delta_h), (0,delta_w)), 'constant', constant_values=(0, 0))
    return new_x

def concatenate_array(ss1): #subjects are supplied in one array combined
    ss = {}
    try:
        ss['xtrain'] = ss1[0]['xtrain']
        ss['ytrain'] = ss1[0]['ytrain']
        for ii in range(1,len(ss1),1):
            ss['xtrain'] = np.concatenate([ss['xtrain'] , ss1[ii]['xtrain']])
            ss['ytrain'] = np.concatenate([ss['ytrain'] , ss1[ii]['ytrain']])
    except:
        ss['xtrain'] = ss1[0][0]['xtrain']
        ss['ytrain'] = ss1[0][0]['ytrain']
        for ii in range(1,len(ss1),1):
            ss['xtrain'] = np.concatenate([ss['xtrain'] , ss1[ii][0]['xtrain']])
            ss['ytrain'] = np.concatenate([ss['ytrain'] , ss1[ii][0]['ytrain']])
    return ss

def concatenate_dicts(ss1): #subjects are supplied in one array combined
    x, y = ss1[0]['xtrain'], ss1[0]['ytrain'] 
    for ii in range(1,len(ss1),1):
        x = np.concatenate([x , ss1[ii]['xtrain']])
        y = np.concatenate([y,  ss1[ii]['ytrain']])
    return x,y

def pad_by_duplicating(x, desired_height=224, desired_width=224): #duplicate signal until the desired new array is full
    x_height, x_width = x.shape[0], x.shape[1]
    new_x = np.zeros((desired_height, desired_width))
    for nhx in range(0, desired_height, x_height):
        for nwx in range(0, desired_width, x_width):
            new_x[nhx:min(nhx+x_height, desired_height), nwx:min(nwx+x_width, desired_width)] = x[0:min(x_height, desired_height-nhx), 0:min(x_width, desired_width-nwx)]
    return new_x 

def generator(x_train, y_train, min_index, max_index, batch_size, desired_size = 299, shuffle = False):
    if max_index is None:
        max_index = len(x_train) - 1
    i = min_index
    
    while 1:
        if shuffle:
            sample_ind = np.random.randint(min_index, max_index, size = batch_size)
            rows = sample_ind
        else:
            if i >= max_index:
                i = min_index
            rows = np.arange(i, min(i + batch_size, max_index))
            i = rows[-1]
        samples = np.zeros((len(rows), desired_size, desired_size))
        targets = np.zeros((len(rows), ))
        for jj, row in enumerate(rows):
            samples[jj] = pad_by_duplicating(x_train[row,:,:], desired_height=desired_size, desired_width=desired_size)
            targets[jj] = y_train[row]
#        samples_with_channel = np.expand_dims(samples, axis=3)
        samples_with_channel = samples[:,:, :, None] * np.ones(3, dtype=int)[None, None, None, :]
        yield samples_with_channel, targets
        
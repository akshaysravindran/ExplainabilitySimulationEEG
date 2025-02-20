"""
    Copyright 2017-2022 Department of Electrical and Computer Engineering
    University of Houston, TX/USA
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
    Please contact Akshay Sujatha Ravindran for more info about licensing asujatharavindran@uh.edu
    via github issues section.
    **********************************************************************************
    Author:     Akshay Sujatha Ravindran
    Date:       08/26/21
    File:       Load the data
    Comments:   This is a helper function to load the files
     
"""

import numpy as np
import mne
import scipy.io as sio
import h5py      


def read_data(filename, iteration=1, siomat_flag=0):
    """
    Function reads the data and channel information
  
    Args:
      filename : name of file
      iteration: version number of file
      siomat_flag: Flag to specify is sio.loadmat is compatible or not
      
  
    Returns:
      X: Data
      Y: output variable
      GT: Ground truth
      fake_evoked: MNE variable
    """
    if siomat_flag:
        Loaded_data          = sio.loadmat(filename) #load data in matlab format   
        # Load the entire set of pre-processed windowed data from Matlab 
        X                    = Loaded_data['X' ] # dimension is samples x time samples x channels
        Y                    = Loaded_data['Y' ].ravel()   # Load the labels  
        GT             = Loaded_data['True_Signal' ] # dimension is samples x time samples x channels
        # X                    = Loaded_data['X' ] # dimension is samples x time samples x channels
        # X= np.moveaxis(X, [0, 1, 2], [0, 2, 1])
        # GT= np.moveaxis(Location, [0, 1, 2], [0, 2, 1]) 
    else:    
        f = h5py.File(filename,'r')    
        X = f['X'][:]   
        
        X= np.moveaxis(X, [0, 1, 2], [2, 1, 0])   
        Y               = f['Y'][:].ravel()        
        Location        = f['True_Signal'][:]
        GT= np.moveaxis(Location, [0, 1, 2], [2, 1, 0])   
    
    
    X= np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))   
    X= X.transpose((0, 3,2,1)) 
    
    
    ch_data=mne.io.read_raw_eeglab('test.set')    
    fake_info = mne.create_info(ch_data.ch_names, sfreq=250.,
                                ch_types='eeg')
    rng = np.random.RandomState(0)
    data = rng.normal(size=(len(ch_data.ch_names), 1)) * 1e-6
    fake_evoked = mne.EvokedArray(data, fake_info)
    fake_evoked.set_montage(ch_data.get_montage())
    
    return X, Y, GT, fake_evoked



def train_validation_test_split(X, Y, Ground_truth, train,valid,test):
    """
    Function reads the data and channel information
  
    Args:
      X : Data
      Y:  True Output
      Ground_truth: Ground Truth explanation  
  
    Returns:
      train test and validation sets
    """
    # Training set
    X_train=X[train]
    Y_train=Y[train]
    GT_train=Ground_truth[train]
    
    
    # Validation set 
    X_valid=X[valid]
    Y_valid=Y[valid]
    GT_valid=Ground_truth[valid]
    
    
    # Test set 
    X_test=X[test]
    Y_test=Y[test]
    GT_test=Ground_truth[test]
    
        
    
    # Make the dimensions compatible for code
    print(np.shape(X_train)) #
    Y_train=np.ravel(Y_train)
    Y_valid=np.ravel(Y_valid)
    Y_test=np.ravel(Y_test)    
    
    
    #%% Shuffle the data
    
    # Shuffle the Training set 
    order = np.arange(len(Y_train))
    np.random.shuffle(order)
    X_train=X_train[order]
    Y_train=Y_train[order]
    GT_train=GT_train[order]
    
    
    # Shuffle the validation set 
    order = np.arange(len(Y_valid))
    np.random.shuffle(order)
    X_valid=X_valid[order]
    Y_valid=Y_valid[order]
    GT_valid=GT_valid[order]
    
    
       
    # Shuffle the Training set 
    order = np.arange(len(Y_test))
    np.random.shuffle(order)
    X_test=X_test[order]
    Y_test=Y_test[order]
    GT_test=GT_test[order]
    
    return X_train, Y_train,  X_valid, Y_valid, X_test, Y_test, GT_train, GT_valid,GT_test

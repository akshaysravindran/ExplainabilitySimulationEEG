
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
    Date:       04/20/21    
    **********************************************************************************
"""


#%%
# Load the libraries and packages required
# Load the libraries and packages required
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import time
import torch.optim as optim
import pandas as pd
from copy import  deepcopy
from captum.attr import DeepLift,LRP,  GuidedBackprop, GuidedGradCam, Deconvolution,InputXGradient,Saliency,LayerGradCam,IntegratedGradients
from dataset_helper import read_data,train_validation_test_split
from models import train_model, validate_model,DeepNet
from pytorchtools import EarlyStopping
from sklearn.metrics import  f1_score, precision_score, recall_score,confusion_matrix
from util import get_random_target
import psutil
from utils_explanation import attribute_image_features,calc__TOPO_measures    
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, FullGrad,LayerCAM
 

#%%


 

# initialization
batch_size                = 128
eeg_chan_num              = 62 # number of channels 
epoch_len                 = 2 # number of epochs to train the model for
pool_dim                  = (1, 2) # Max Pooling dimension
strides_pool              = (1, 2) # Stride dimension for pool layers
padding                   = 'valid' # Type of zero padding (currently set to not pad)
num_units                 = 32 # Number of convolutional layers units per layer
# ITERATE_list              = (5,7.5,2.5)
ITERATE_list              = (15,10,7.5,5,2.5)
conv_size                 = 5
iteration                 = 3 
split_percentage          = (0.6, 0.2,0.2) # Train %, validation %, test %
device                    = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from sklearn.model_selection import KFold 
indices= np.array(range(1,20000,1))
kf = KFold(n_splits=5)
fold=0
for train_index, test_index in kf.split(indices):
    fold=fold+1
    for iteration in ITERATE_list:
    # Files
        filename                  = 'Spatial_1_%g.mat'%iteration 
        BW_Tall                   = '%s_weights_%d.pt'%(filename[:-4],fold)
        file = 'Spatial/Data/'+filename
        # file = filename
        # Use GPU if available
        
        # Load the data
        X, Y, GT, fake_evoked=read_data(file, iteration, siomat_flag=1)        
        
        # Divide the data into training, validation and test set
        X_train, Y_train,  X_valid, Y_valid, X_test, Y_test, GT_train, GT_valid,GT_test = train_validation_test_split(X, Y, GT,
                                                                      train_index, test_index[:len(test_index)//2], test_index[len(test_index)//2:] )
        
        num_classes = len(np.unique(Y_valid)) # number of classes to decode
        print(num_classes)
        del X, Y, GT
        # %% Convert to tensors
        tensor_xtrain = torch.Tensor(X_train) #transform to torch tensor
        tensor_ytrain = torch.tensor(Y_train, dtype=torch.long)
        
        tensor_xvalid = torch.Tensor(X_valid) #transform to torch tensor
        tensor_yvalid = torch.tensor(Y_valid, dtype=torch.long)
        
        
        tensor_xtest = torch.Tensor(X_test) #transform to torch tensor
        tensor_ytest = torch.tensor(Y_test, dtype=torch.long)
        tensor_gttest = torch.Tensor(GT_test) #transform to torch tensor
        
        
        # create your training dataloader
        train_dataset = TensorDataset(tensor_xtrain,tensor_ytrain) # create your datset
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle =True) 
        
        # create your validation dataloader
        valid_dataset = TensorDataset(tensor_xvalid,tensor_yvalid) # create your datset
        valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle =True) 
        
        # create your test dataloader
        test_dataset = TensorDataset(tensor_xtest,tensor_ytest,tensor_gttest) # create your datset
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle =True) # create your dataloader
        del X_train,Y_train,X_valid,Y_valid,X_test,Y_test,GT_train,GT_valid,GT_test
        
        
        del tensor_xtrain,tensor_ytrain,tensor_xvalid,tensor_yvalid,tensor_xtest,tensor_gttest,tensor_ytest
        
        
        #%% Define the model
    
    
        model = DeepNet(num_units, conv_size, eeg_chan_num,pool_dim,strides_pool,num_classes).to(device)
        print(model)
        
        # Training parameters
        criterion                 = nn.CrossEntropyLoss()
        optimizer                 = optim.Adam(model.parameters(), lr=0.0001)
        early_stopping            = EarlyStopping(patience=5, verbose=True, path =BW_Tall)
        epochs                    = 500
        train_loss,train_accuracy = [], []
        val_loss, val_accuracy    = [], []
        
        
        # Train and evaluate the model
        start = time.time()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} of {epochs}")
    
            # Train the model foe an epoch    
            model=train_model(model, train_dataloader, optimizer, criterion, device)
                              
                    
            train_epoch_loss, train_epoch_accuracy = validate_model(
                model, train_dataloader, criterion, device)
            
            # Compute the validation loss and accuracy
            val_epoch_loss, val_epoch_accuracy = validate_model(
                model, valid_dataloader, criterion, device)
            
            # Save the history
            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_accuracy)
            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)
            
            # Stop training if it does not improve in patience number of epochs
            early_stopping(val_epoch_loss, model)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break    
            print(f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_accuracy:.2f}")
            print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
        model.load_state_dict(torch.load(BW_Tall))    
        
        val_epoch_loss, val_epoch_accuracy = validate_model(
            model, valid_dataloader, criterion, device)
        
        train_epoch_loss, train_epoch_accuracy = validate_model(
            model, train_dataloader, criterion, device)
            
        
        print(f'Final- Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_accuracy:.2f}')
        
        end = time.time()
        print(f"Training time: {(end-start)/60:.3f} minutes")    
        del train_dataloader,train_dataset, valid_dataloader, valid_dataset    
    
    #%%
    
    
        torch.cuda.empty_cache() 
        # num_classes=2
        dataiter = iter(test_dataloader)    
        Performance_Randomized = pd.DataFrame()
        Performance_Weight_Randomized = pd.DataFrame()    
        Performance_SpatialRandomized = pd.DataFrame()
        Performance_Weight_Randomized_Spatial= pd.DataFrame()
        PERF=[]
        for images, labels, true_loc in dataiter:   
        
            output    = model(images.to(device))
            pred      = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            locate    = (pred.eq(labels.to(device).view_as(pred)).cpu().detach().numpy().squeeze())
            
            test_acc  = 100. *np.shape(np.where(locate))[1]/ np.shape((locate))[0]
            f_score=(f1_score(labels.cpu().detach().numpy().squeeze(), pred.cpu().detach().numpy().squeeze(), average="macro"))
            p_score=(precision_score(labels.cpu().detach().numpy().squeeze(), pred.cpu().detach().numpy().squeeze(), average="macro"))
            r_score=(recall_score(labels.cpu().detach().numpy().squeeze(), pred.cpu().detach().numpy().squeeze(), average="macro"))
            cm=(confusion_matrix(labels.cpu().detach().numpy().squeeze(), pred.cpu().detach().numpy().squeeze()))
            print(test_acc)
            PERF.append([test_acc,f_score,p_score,r_score,cm])
            del pred, output,test_acc, f_score, p_score, r_score, cm
            
            images    = images[locate].to(device)
            labels    = labels[locate].to(device)
            locations = true_loc[locate]
            locations = locations.cpu().detach().numpy().squeeze().transpose((0,2,1))
            
            labelsnpy = labels.cpu().detach().numpy().squeeze()  
            # Find incorrect label
            rand_label=np.zeros(np.shape(labelsnpy),dtype=np.int64)
            for i in range(len(labelsnpy)):        
                rand_label[i]=get_random_target(labelsnpy[i],num_classes)    
            input                           = images
            input.requires_grad             = True   
            
        
        
            df_S                              = pd.DataFrame()
            df                                = pd.DataFrame()
            for i in range(8):
                g_cam_flag = False
                if i==0:
                    gradcam = Saliency(model)
                    name= 'Saliency'
                elif i==1:     
                     gradcam = Deconvolution(model)
                     name= 'Deconvolution'
                elif i==2:       
                    gradcam = GuidedBackprop(model)
                    name= 'GuidedBackprop'
                elif i==3:      
                    gradcam = GuidedGradCam(model, model.conv5)
                    name= 'GuidedGradCam'
                elif i==4:       
                    gradcam = DeepLift(model)
                    name= 'DeepLift'
                elif i==5:        
                    gradcam = LRP(model)
                    name= 'LRP'
                elif i==6:
                    gradcam = InputXGradient(model)
                    name= 'InputXGradient'
                elif i==7:
                    gradcam = LayerGradCam(model, model.conv5)
                    name= 'GradCAM'
                    g_cam_flag = True  
                elif i==8:
                    gradcam = IntegratedGradients(model)
                    name= 'IG'
                    g_cam_flag = False                      
                    
                    
                grad                        = attribute_image_features(gradcam, model, input,labels, g_cam_flag=g_cam_flag)  
                gradrand                    = attribute_image_features(gradcam, model, input,torch.tensor(rand_label, dtype=torch.long).to(device), g_cam_flag=g_cam_flag)         
                
                
                Measures_spatial, count             = calc__TOPO_measures(grad, locations ,fake_evoked, gt_flag=1, abs_condition = 0, prctile_val=95)
                Measuresrand_spatial, countrand     = calc__TOPO_measures(gradrand, grad,fake_evoked, gt_flag=0, abs_condition = 0,  prctile_val=95)            
                
                
                df2                         = {(name+'-TrueS'):np.mean(Measures_spatial,axis=0),(name+'-RandS'):np.mean(Measuresrand_spatial,axis=0), (name+'nanS'):count}            
                df2                         = pd.DataFrame(df2)              
                df_S                          = pd.concat([df_S,df2],axis=1) 
                print(name)
                del gradcam,grad,gradrand,Measures_spatial,Measuresrand_spatial
                torch.cuda.empty_cache()
                
                
            for i in range(5):
                if i==0:
                    cam = GradCAM(model=model, target_layers=[model.conv5],use_cuda=True)
                    name= 'GradCAM2'
                elif i==1:
                    cam = GradCAMPlusPlus(model=model, target_layers=[model.conv5],use_cuda=True)
                    name ='GradCAMPlusPlus'
                    
                elif i==2:
                    cam = ScoreCAM(model=model,target_layers=[model.conv5],use_cuda=True)
    
                    name='ScoreCAM'  
                elif i==3:
                    cam =  FullGrad(model=model,target_layers=[model.conv5],use_cuda=True)
                    name=' FullGrad' 
                elif i==4:
                    cam =  LayerCAM(model=model,target_layers=[model.conv5],use_cuda=True)
                    name=' LayerCAM' 
    
    
                grad                        = cam(input_tensor=input, target_category=labels)
                gradrand                    = cam(input_tensor=input, target_category=torch.tensor(rand_label, dtype=torch.long).to(device))
                Measures_spatial, count             = calc__TOPO_measures(grad, locations ,fake_evoked, gt_flag=1, abs_condition = 0, prctile_val=95)
                Measuresrand_spatial, countrand     = calc__TOPO_measures(gradrand, grad,fake_evoked, gt_flag=0, abs_condition = 0,  prctile_val=95)  
                df2                         = {(name+'-TrueS'):np.mean(Measures_spatial,axis=0),(name+'-RandS'):np.mean(Measuresrand_spatial,axis=0), (name+'nanS'):count}   
               
                df2                         = pd.DataFrame(df2)  
                
                df_S                        = pd.concat([df_S,df2],axis=1) 
                del cam,grad,gradrand,Measures_spatial,Measuresrand_spatial
                torch.cuda.empty_cache()
                print(name)
            Performance_Randomized                               = Performance_Randomized.append(df_S)
            del df_S
            
            
            

            
            
            model_randomize = deepcopy(model)
        
            
            
            Layer_list=[]
            for layer in model_randomize.children():
                if hasattr(layer, 'reset_parameters'):
                    Layer_list.append(layer)
            df_S                            = pd.DataFrame()               
            df                              = pd.DataFrame()       
            for layer in reversed(Layer_list):
                layer.reset_parameters()    
                for i in range(8):  
                        g_cam_flag = False
                        if i==0:
                            gradcam_orig = Saliency(model)
                            gradcam_shuffled = Saliency(model_randomize)
                            name= 'Saliency'
                        elif i==1:     
                             gradcam_orig = Deconvolution(model)
                             gradcam_shuffled = Deconvolution(model_randomize)
                             name= 'Deconvolution'
                        elif i==2:       
                            gradcam_orig = GuidedBackprop(model)
                            gradcam_shuffled = GuidedBackprop(model_randomize)
                            name= 'GuidedBackprop'
                        elif i==3:      
                            gradcam_orig = GuidedGradCam(model, model.conv5)
                            gradcam_shuffled = GuidedGradCam(model_randomize, model_randomize.conv5)
                            name= 'GuidedGradCam'
                        elif i==4:       
                            gradcam_orig = DeepLift(model)
                            gradcam_shuffled = DeepLift(model_randomize)
                            name= 'DeepLift'
                        elif i==5:        
                            gradcam_orig = LRP(model)
                            gradcam_shuffled = LRP(model_randomize)
                            name= 'LRP'
                        elif i==6:
                            gradcam_orig = InputXGradient(model)
                            gradcam_shuffled = InputXGradient(model_randomize)
                            name= 'InputXGradient'
                        elif i==7:
                            gradcam_orig = LayerGradCam(model, model.conv5)
                            gradcam_shuffled = LayerGradCam(model_randomize, model_randomize.conv5)
                            name= 'GradCAM'
                            g_cam_flag = True
                        elif i==8:
                            gradcam_orig = IntegratedGradients(model)
                            gradcam_shuffled = IntegratedGradients(model_randomize)
                            name= 'IG'
                            g_cam_flag = False                             
                            
                        grad                        = attribute_image_features(gradcam_orig, model, input,labels, g_cam_flag=g_cam_flag)
                        gradrand                    = attribute_image_features(gradcam_shuffled, model_randomize, input,labels, g_cam_flag=g_cam_flag)   
                        Measures_spatialrand, count             = calc__TOPO_measures(gradrand, grad ,fake_evoked, gt_flag=0, abs_condition = 0, prctile_val=95) 
                        df2={(str(layer)[:6] + name +'S' ):np.mean(Measures_spatialrand,axis=0)}    
                        df2                         = pd.DataFrame(df2)    
                        df_S                          = pd.concat([df_S,df2],axis=1)     
                        print(name)
                        del gradcam_orig,grad,gradrand,gradcam_shuffled,Measures_spatialrand,df2
                        torch.cuda.empty_cache()
                        
                    

                        
                for i in range(5):
                        if i==0:
                            cam      = GradCAM(model=model, target_layers=[model.conv5],use_cuda=True)
                            cam_rand = GradCAM(model=model_randomize, target_layers=[model_randomize.conv5],use_cuda=True)
                            name= 'GradCAM2'
                        elif i==1:
                            cam = GradCAMPlusPlus(model=model, target_layers=[model.conv5],use_cuda=True)
                            cam_rand = GradCAMPlusPlus(model=model_randomize, target_layers=[model_randomize.conv5],use_cuda=True)
                            name ='GradCAMPlusPlus'
                        elif i==2:
                            cam = ScoreCAM(model=model, target_layers=[model.conv5],use_cuda=True)
                            cam_rand = ScoreCAM(model=model_randomize, target_layers=[model_randomize.conv5],use_cuda=True)
                            name='ScoreCAM'  
                        elif i==3:
                            cam =  FullGrad(model=model, target_layers=[model.conv5],use_cuda=True)
                            cam_rand =  FullGrad(model=model_randomize, target_layers=[model_randomize.conv5],use_cuda=True)
                            name=' FullGrad'   
                        elif i==4:
                            cam =  LayerCAM(model=model, target_layers=[model.conv5],use_cuda=True)
                            cam_rand =  LayerCAM(model=model_randomize, target_layers=[model_randomize.conv5],use_cuda=True)
                            name=' LayerCAM'                                  
                                                    
                            
                            
                            
                            
                        grad                        = cam(input_tensor=input, target_category=labels)
                        gradrand                    = cam_rand(input_tensor=input, target_category=labels)
                        Measures_spatialrand, count             = calc__TOPO_measures(gradrand, grad ,fake_evoked, gt_flag=0, abs_condition = 0, prctile_val=95)
                        df2={(str(layer)[:6] + name +'S' ):np.mean(Measures_spatialrand,axis=0)}    
                        df2                         = pd.DataFrame(df2)    
                        df_S                          = pd.concat([df_S,df2],axis=1)    
                        del cam,grad,gradrand,cam_rand,Measures_spatialrand
                        torch.cuda.empty_cache()
                        print(name)
            Performance_Weight_Randomized                        = Performance_Weight_Randomized.append(df_S)        
            print('Randomized weight pytorch-grad-cam')
            print(psutil.cpu_percent())
            print(psutil.virtual_memory())  # physical memory usage
            print('memory % used:', psutil.virtual_memory()[2])    
            
            
            
            del images, labels, locations, input,rand_label, model_randomize,df_S
            
            
        np.savez('%sv_%d.npz'%(BW_Tall[:-11],fold), Performance_Weight_Randomized=Performance_Weight_Randomized,
                                               Performance_Randomized=Performance_Randomized,                                     
                                               PERF=np.array(PERF),train_epoch_accuracy=train_epoch_accuracy, 
                                               val_epoch_accuracy=val_epoch_accuracy,                                    
                                               allow_pickle=True)     
        # Performance_Weight_Randomized.to_pickle( '%sv1'%(BW_Tall[:-11])+ "Weight_Random.pkl")
        # Performance_Randomized.to_pickle( '%sv1'%(BW_Tall[:-11])+ "Label_Random.pkl")
        
     
        
        
        del Performance_Weight_Randomized,Performance_Randomized, PERF,dataiter, test_dataset,Performance_SpatialRandomized,Performance_Weight_Randomized_Spatial
    
    

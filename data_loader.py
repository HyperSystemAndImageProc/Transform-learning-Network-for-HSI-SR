from os import listdir
from torch.nn import functional as F
import cv2
import torch
import numpy as np
import os
import random
import scipy.io as scio
import h5py


def build_datasets(root, dataset, size, n_select_bands, scale_ratio):
    # Image preprocessing, normalization for the pretrained resnet
    if dataset == 'Pavia':
        img = scio.loadmat(root + '/' + 'Pavia.mat')['HSI_original']*1.0
        # img = scio.loadmat(root + '/' + 'Pavia_RCRS.mat')['x'] * 1.0 
    elif dataset == 'PaviaU':
        img = scio.loadmat(root + '/' + 'PaviaU.mat')['paviaU']*1.0
        # img = scio.loadmat(root + '/' + 'PaviaU_RCRS.mat')['x'] * 1.0
    elif dataset == 'Botswana':
        img = scio.loadmat(root + '/' + 'Botswana.mat')['Botswana']*1.0
        # img = scio.loadmat(root + '/' + 'Botswana_RCRS.mat')['x'] * 1.0
    elif dataset == 'IndianP':
        img = scio.loadmat(root + '/' + 'Indian_pines.mat')['indian_pines_corrected']*1.0
        # img = scio.loadmat(root + '/' + 'Indian_pines_RCRS.mat')['x'] * 1.0 
    elif dataset == 'Washington':
        img = scio.loadmat(root + '/' + 'Washington_DC.mat')['DC']*1.0
        # img = scio.loadmat(root + '/' + 'Washington_DC_RCRS.mat')['x'] * 1.0
     elif dataset == 'Trento':
        img = scio.loadmat(root + '/' + 'HSI_Trento.mat')['hsi_trento']*1.0
        # img = scio.loadmat(root + '/' + 'Trento_RCRS.mat')['x'] * 1.0

    print (img.shape)
    max = np.max(img)
    min = np.min(img)
    img = 255*((img - min) / (max - min + 0.0)) 

    # throwing up the edge
    w_edge = img.shape[0]//scale_ratio*scale_ratio-img.shape[0]
    h_edge = img.shape[1]//scale_ratio*scale_ratio-img.shape[1]
    w_edge = -1  if w_edge==0  else  w_edge
    h_edge = -1  if h_edge==0  else  h_edge
    img = img[:w_edge, :h_edge, :]

    # cropping area 
    width, height, n_bands = img.shape 
    w_str = (width - size) // 2 
    h_str = (height - size) // 2 
    w_end = w_str + size
    h_end = h_str + size
    img_copy = img.copy()

    # test sample 
    gap_bands = n_bands / (n_select_bands-1.0)
    test_ref = img_copy[w_str:w_end, h_str:h_end, :].copy() #128，128，n_band
    test_lr = cv2.GaussianBlur(test_ref, (5,5), 2) 
    test_lr = cv2.resize(test_lr, (size//scale_ratio, size//scale_ratio)) 
    test_hr = test_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):
        test_hr = np.concatenate((test_hr, test_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    test_hr = np.concatenate((test_hr, test_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)

    # training sample
    img[w_str:w_end,h_str:h_end,:] = 0
    train_ref = img
    train_lr = cv2.GaussianBlur(train_ref, (5,5), 2)##
    train_lr = cv2.resize(train_lr, (train_lr.shape[1]//scale_ratio, train_lr.shape[0]//scale_ratio))  
    train_hr = train_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1): 
        train_hr = np.concatenate((train_hr, train_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    train_hr = np.concatenate((train_hr, train_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)


    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0) 
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0) 
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0) 
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0) 
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0) 
   # ref = torch.from_numpy(img_copy).permute(2,0,1).unsqueeze(dim=0) 
    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]
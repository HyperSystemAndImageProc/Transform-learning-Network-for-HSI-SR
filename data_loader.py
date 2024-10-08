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
        img = scio.loadmat(root + '/' + 'Pavia.mat')['HSI_original']*1.0#.loadmat结果是字典数据 方括号是最后一项的名字
        # img = scio.loadmat(root + '/' + 'pretrain_half_P.mat')['x'] * 1.0  #空间无光通道相关
    elif dataset == 'PaviaU':
        img = scio.loadmat(root + '/' + 'PaviaU.mat')['paviaU']*1.0#先运行不带[]的 确定最后一项名字
        # img = scio.loadmat(root + '/' + 'PaviaU.mat')  #原始
        # img = scio.loadmat(root + '/' + 'pretrain_samespace.mat')['x'] * 1.0 #空间相同 通道相关 RCFS
        # img = scio.loadmat(root + '/' + 'PaviaU_pre_space.mat')['x'] * 1.0 #空间相关 通道无关 RSRC
        # img = scio.loadmat(root + '/' + 'pretrain_all1half.mat')['x'] * 1.0  #空间无光通道相关 !! RcRs
        # img = scio.loadmat(root + '/' + 'PaviaU_pre_chanane_same.mat')['x'] * 1.0   #通道相同 RSFC
        # img = scio.loadmat(root + '/' + 'pretrain_samespacenew.mat')['x'] * 1.0  # 空间相同 通道相关 4个像素测试
    elif dataset == 'Botswana':
        img = scio.loadmat(root + '/' + 'Botswana.mat')['Botswana']*1.0
        # img = scio.loadmat(root + '/' + 'pretrain_half_Bo.mat')['x'] * 1.0  # 空间无光通道相关
        # img = scio.loadmat(root + '/' + 'Botswana_RCFS.mat')['x'] * 1.0  #RCFS
        # img = scio.loadmat(root + '/' + 'Botswana_RCRS.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + 'Botswana_RSFC.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + 'Botswana_RSRC.mat')['x'] * 1.0
    elif dataset == 'KSC':
        img = scio.loadmat(root + '/' + 'KSC.mat')['KSC']*1.0
    elif dataset == 'IndianP':
        img = scio.loadmat(root + '/' + 'Indian_pines.mat')['indian_pines_corrected']*1.0
        # img = scio.loadmat(root + '/' + 'pretrain_half_IP.mat')['x'] * 1.0  #空间无光通道相关
    elif dataset == 'Washington':
        img = scio.loadmat(root + '/' + 'Washington_DC.mat')['DC']*1.0
        # img = scio.loadmat(root + '/' + 'pretrain_half_WA.mat')['x'] * 1.0  # 空间无光通道相关
    elif dataset == 'Urban':
        img = scio.loadmat(root + '/' + 'Urban.mat')['Y']
        img = np.reshape(img, (162, 307, 307))*1.0
        img = np.swapaxes(img, 0,2)
    elif dataset == 'Trento':
        img = scio.loadmat(root + '/' + 'HSI_Trento.mat')['hsi_trento']*1.0
        # img = scio.loadmat(root + '/' + 'Trento_RCFS.mat')['x'] * 1.0  #RCFS
        # img = scio.loadmat(root + '/' + 'Trento_RCRS.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + 'Trento_RSFC.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + 'Trento_RSRC.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '2pix.mat')['x'] * 1.0  #RCFS
        # img = scio.loadmat(root + '/' + '4pix.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '8pix.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '16pix.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '32pix.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '64pix.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '128pix.mat')['x'] * 1.0
        # img = scio.loadmat(root + '/' + '1fig.mat')['data_cube'] * 1.0
        # img = scio.loadmat(root + '/' + '2fig.mat')['data_cube'] * 1.0
        # img = scio.loadmat(root + '/' + '4fig.mat')['data_cube'] * 1.0
        # img = scio.loadmat(root + '/' + '8fig.mat')['data_cube'] * 1.0
        # img = scio.loadmat(root + '/' + '16fig.mat')['data_cube'] * 1.0
        # img = scio.loadmat(root + '/' + '32fig.mat')['data_cube'] * 1.0
        # img = scio.loadmat(root + '/' + '48fig.mat')['data_cube'] * 1.0
    print (img.shape)
    max = np.max(img)
    min = np.min(img)
    img = 255*((img - min) / (max - min + 0.0)) #归一化

    # throwing up the edge
    w_edge = img.shape[0]//scale_ratio*scale_ratio-img.shape[0]
    h_edge = img.shape[1]//scale_ratio*scale_ratio-img.shape[1]
    w_edge = -1  if w_edge==0  else  w_edge
    h_edge = -1  if h_edge==0  else  h_edge
    img = img[:w_edge, :h_edge, :]

    # cropping area 选取训练位置
    width, height, n_bands = img.shape 
    w_str = (width - size) // 2 
    h_str = (height - size) // 2 
    w_end = w_str + size
    h_end = h_str + size
    img_copy = img.copy()

    # test sample MSI退化
    gap_bands = n_bands / (n_select_bands-1.0)
    test_ref = img_copy[w_str:w_end, h_str:h_end, :].copy() #128，128，n_band
    test_lr = cv2.GaussianBlur(test_ref, (5,5), 2) #高斯矩阵的长宽为5标准差取2
    test_lr = cv2.resize(test_lr, (size//scale_ratio, size//scale_ratio)) #双线性插值下采样
    #hr 搁一定波段选取一个
    test_hr = test_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):
        test_hr = np.concatenate((test_hr, test_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    test_hr = np.concatenate((test_hr, test_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)

    # training sample
    img[w_str:w_end,h_str:h_end,:] = 0
    train_ref = img
    train_lr = cv2.GaussianBlur(train_ref, (5,5), 2)##
    train_lr = cv2.resize(train_lr, (train_lr.shape[1]//scale_ratio, train_lr.shape[0]//scale_ratio))  #原图下采样4倍
    train_hr = train_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):  #训练的hr原图选波段
        train_hr = np.concatenate((train_hr, train_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    train_hr = np.concatenate((train_hr, train_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)


    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0) 
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0) 
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0) 
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0) 
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0) 
   # ref = torch.from_numpy(img_copy).permute(2,0,1).unsqueeze(dim=0) #####原图
    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]
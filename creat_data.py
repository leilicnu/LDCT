# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:13:49 2021

@author: lml
"""

import pydicom
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def extrartimage(data_dir, verbose=False):
    file_list = glob.glob(data_dir+'/*.dcm')
    data = []
    for i in range(len(file_list)):
        dcm = pydicom.read_file(file_list[i])
        img_arr = dcm.pixel_array
        img_arr = np.array(img_arr)
        #im_arr = np.log(img_arr+1)#对数变换
        img_arr = 255.0*(img_arr- np.min(img_arr)) / (np.max(img_arr)- np.min(img_arr))#归一化[0,1]
        data.append(img_arr)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='float32')#tensor不支持uint16
    return data

def show(x, title=None, cbar=False, figsize=None):
    if type(x) is torch.Tensor:
        x = x[0]
    plt.figure(figsize=figsize)
    plt.imshow(x, cmap=plt.cm.bone)
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
    
def get_inputs(data_dir, train = True, verbose=False):
    """

    Parameters
    ----------
    data_dir : string
    train : bool, optional
    verbose : bool, optional

    Returns
    -------
    inputs : np.ndarry or tensor(if train is False)

    """
    data_dir_list = glob.glob(data_dir+'/*')
    inputs = extrartimage(data_dir_list[0]+'\\input', verbose)
    for da_dir in data_dir_list[1:]:
        image = extrartimage(da_dir+'\\input', verbose)
        inputs = np.vstack((inputs, image))
    if not train:
        inputs = torch.from_numpy(inputs).unsqueeze(dim=1)
    return inputs
    
def get_targets(data_dir, train = True, verbose=False):
    data_dir_list = glob.glob(data_dir+'/*')
    targets = extrartimage(data_dir_list[0]+'\\target', verbose)
    for da_dir in data_dir_list[1:]:
        image = extrartimage(da_dir+'\\input', verbose)
        targets = np.vstack((targets, image))
    if not train:
        targets = torch.from_numpy(targets).unsqueeze(dim=1)
    return targets
    
def get_patches(img, patch_size = 64, stride = 10):
    h, w = img.shape
    patches = []
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size]
            patches.append(x)
    return patches


def get_traindata(inputs, targets, patch_size = 64, patch_stride = 30, verbose = False):
    data_in = []
    data_ta = []
    for i in range(len(inputs)):
        patches_in = get_patches(inputs[i], patch_size, patch_stride)
        patches_ta = get_patches(targets[i], patch_size, patch_stride)
        for j in range(len(patches_in)):
            data_in.append(patches_in[j])
            data_ta.append(patches_ta[j])
        if verbose:
            print(str(i+1) + '/' + str(len(inputs)) + ' is done ^_^')
    data_in = np.array(data_in)
    data_in = np.expand_dims(data_in, axis = 1)
    data_in = torch.from_numpy(data_in)
    data_ta = np.array(data_ta)
    data_ta = np.expand_dims(data_ta, axis = 1)
    data_ta = torch.from_numpy(data_ta)
                 
    print('^_^-training data finished-^_^')
    return (data_in, data_ta)

    
            

if __name__ == '__main__': 
    data_dir =  "C:\\Users\\Team1\\lml\\LDCT\\Data\\Mayo_chest\\trainset"
    inputs = get_inputs(data_dir, train = True, verbose = False)
    targets = get_targets(data_dir, train = True, verbose = False)
    print(inputs.shape)
    show(inputs[0], cbar = True)
    show(targets[0], cbar = True)
    
    data_in, data_ta = get_traindata(inputs, targets, patch_size = 64,
                                     patch_stride = 30, verbose = False)
    show(data_in[100], cbar = True)




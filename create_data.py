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
import random

def extrartimage(data_dir, verbose=False):
    file_list = glob.glob(data_dir+'/*.dcm')
    data = []
    for i in range(len(file_list)):
        dcm = pydicom.read_file(file_list[i])
        img_arr = dcm.pixel_array
        img_arr = np.array(img_arr)
        #im_arr = np.log(img_arr+1)#对数变换
        #im_arr = im_log/np.max(im_arr)#归一化
        data.append(img_arr)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='float32')#tensor不支持uint16
    return data

def show(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(x, cmap=plt.cm.bone)
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()
    
def data_gain(data_dir):
    data_dir_list = glob.glob(data_dir+'/*')
    inputs = extrartimage(data_dir_list[0]+'\\input')
    targets = extrartimage(data_dir_list[0]+'\\target')
    for da_dir in data_dir_list[1:]:
        image = extrartimage(da_dir+'\\input')
        target = extrartimage(da_dir+'\\target')
        inputs = np.vstack((inputs, image))
        targets= np.vstack((targets, target))     
    return (inputs, targets)

def div_train_val(data_dir, ratio = .2):
    inputs, targets = data_gain(data_dir)
    total_num = len(inputs)
    val_num = int(total_num * ratio)
    l = list(np.linspace(0,total_num-1,total_num, dtype = 'int'))
    random.shuffle(l)

    trainset_input = inputs[l[0:total_num - val_num]]
    trainset_target = targets[l[0:total_num - val_num]]
    valset_input = inputs[l[total_num - val_num:]]
    valset_target = targets[l[total_num - val_num:]]
    return (trainset_input, trainset_target, valset_input, valset_target)
    

def get_patches(img, patch_size = 64, stride = 10):
    h, w = img.shape
    patches = []
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size]
            patches.append(x)
    return patches


def datagenerator(trainset_input, trainset_target, batch_size = 128, 
                  patch_size = 64, patch_stride = 30, ratio = .8, verbose=False):

    data_in = []
    data_ta = []
    for i in range(len(trainset_input)):
        patches_in = get_patches(trainset_input[i], patch_size, patch_stride)
        patches_ta = get_patches(trainset_target[i], patch_size, patch_stride)
        for j in range(len(patches_in)):
            if (np.sum(patches_in[j]<10)/patch_size**2 < ratio):
                #空气占比超过ratio,就不要了
                data_in.append(patches_in[j])
                data_ta.append(patches_ta[j])
        if verbose:
            print(str(i+1) + '/' + str(len(trainset_input)) + ' is done ^_^')

    data_in = np.array(data_in)
    data_in = np.expand_dims(data_in, axis = 1)
    data_in = torch.from_numpy(data_in)
                              
    data_ta = np.array(data_ta)
    data_ta = np.expand_dims(data_ta, axis = 1)
    data_ta = torch.from_numpy(data_ta)
                 
    print('^_^-training data finished-^_^')
    return (data_in, data_ta)
    data_in, data_ta = datagenerator(trainset_input, trainset_target, batch_size = 128, 
                  patch_size = 64, patch_stride = 35, ratio = .4, verbose=False)
            

if __name__ == '__main__': 
    data_dir =  "F:\BaiduNetdiskDownload\LDCT_ImageData\SV_50mAs"

    inputs , targets = data_gain(data_dir)

    show(inputs[0][0], cbar = True, figsize=(15,12))
    show(targets[0][0], cbar = True, figsize=(15,12))





'''
import matplotlib.image as mpimg
info = {}
# 读取dicom文件
dcm = pydicom.read_file(r"F:\BaiduNetdiskDownload\LDCT_ImageData\SV_50mAs\C3710\input\000.dcm")
# 通过字典关键字来获取图像的数据元信息（当然也可以根据TAG号）
# 这里获取几种常用信息
info["PatientID"] = dcm.PatientID               # 患者ID
info["PatientName"] = dcm.PatientName           # 患者姓名
#info["PatientBirthData"] = dcm.PatientBirthData # 患者出生日期
info["PatientAge"] = dcm.PatientAge             # 患者年龄
info['PatientSex'] = dcm.PatientSex             # 患者性别
info['StudyID'] = dcm.StudyID                   # 检查ID
info['StudyDate'] = dcm.StudyDate               # 检查日期
info['StudyTime'] = dcm.StudyTime               # 检查时间
info['InstitutionName'] = dcm.InstitutionName   # 机构名称
info['Manufacturer'] = dcm.Manufacturer         # 设备制造商
info['StudyDescription']=dcm.StudyDescription   # 检查项目描述
print(info)

filename = r"F:\BaiduNetdiskDownload\LDCT_ImageData\SV_50mAs\C3710\input\000.dcm"
pngname =  r"E:\li\Desktop\首都师范大学\课程\learn\LDCT\LDCT_ImageData\SV_50mAs\C3710\input\000.png"
# 读取dicom文件
dcm = pydicom.read_file(filename)
# 获取图像唯一标识符UID
uid = dcm.SOPInstanceUID
# 获取像素矩阵
img_arr = dcm.pixel_array
# 打印矩阵大小
print(img_arr.shape)
# 获取像素点个数
lens = img_arr.shape[0]*img_arr.shape[1]
# 获取像素点的最大值和最小值
arr_temp = np.reshape(img_arr,(lens,))
max_val = max(arr_temp)
min_val = min(arr_temp)
# 图像归一化
img_arr = (img_arr-min_val)/(max_val-min_val)
# 绘制图像并保存
plt.figure(figsize=(12,12),dpi=250) # 图像大小可以根据文件不同进行设置
plt.imshow(img_arr,cmap=plt.cm.bone)
plt.title("UID:{}".format(uid))
mpimg.imsave(pngname,img_arr)
plt.close()


from PIL import Image
im = Image.fromarray(img_arr)
im.convert('L').save(pngname)
im = plt.imread('000.png')
plt.imshow(im,cmap=plt.cm.bone)
plt.imshow(im-img_arr,cmap=plt.cm.bone)


'''







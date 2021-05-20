import glob
import pydicom
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, args):
        # dicom file dir
        self.data_dir_list = glob.glob(args.data_path + '/*')
        self.print_progress = args.print_progress
        # image params
        self.patch_size = args.patch_size
        self.patch_stride = args.patch_stride
        self.air_ratio = args.air_ratio
        self.data_range = args.data_range
        # training params
        self.train = args.train
        # get image
        ldct_path_list = []
        ndct_path_list = []
        for data_dir in self.data_dir_list:
            ldct_path_list.append(data_dir + '\\input')
            ndct_path_list.append(data_dir + '\\target')
        ldct_images = self.extrartimage(ldct_path_list)
        ndct_images = self.extrartimage(ndct_path_list)
        if self.train:
            self.ldct_images, self.ndct_images = self.get_patches(ldct_images, ndct_images, args)
        else:
            ldct_images = np.array(ldct_images)
            ldct_images = torch.from_numpy(ldct_images)
            self.ldct_images = ldct_images.unsqueeze(dim=1)

            ndct_images = np.array(ndct_images)
            ndct_images = torch.from_numpy(ndct_images)
            self.ndct_images = ndct_images.unsqueeze(dim=1)

    def __len__(self):
        return len(self.ldct_images)

    def __getitem__(self, idx):
        return self.ldct_images[idx], self.ndct_images[idx]

    def extrartimage(self, path_list):
        image_list = []
        for path in path_list:
            file_list = glob.glob(path + '/*.dcm')
            for file in file_list:
                dcm = pydicom.read_file(file)
                img_arr = dcm.pixel_array
                img_arr = np.array(img_arr, dtype='float32')
                # img_arr = np.log(img_arr+1)#对数变换
                img_arr = self.data_range * ((img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr)))
                image_list.append(img_arr)
                if self.print_progress:
                    print(file + " is done")
        return image_list

    def get_patches(self, ldct_images, ndct_images, args):
        h, w = ldct_images[0].shape
        patch_size = self.patch_size
        patch_stride = self.patch_stride
        air_num = 0.3 * patch_size ** 2  # 空气占比30%
        ldct_patches = []
        ndct_patches = []
        for n in range(len(ldct_images)):
            for i in range(0, h - patch_size + 1, patch_stride):
                for j in range(0, w - patch_size + 1, patch_stride):
                    p_ldct = ldct_images[n][i:i + patch_size, j:j + patch_size]
                    p_ldct = self.data_range * (p_ldct - np.min(p_ldct)) / (np.max(p_ldct) - np.min(p_ldct))
                    p_ndct = ndct_images[n][i:i + patch_size, j:j + patch_size]
                    p_ndct = self.data_range * (p_ndct - np.min(p_ndct)) / (np.max(p_ndct) - np.min(p_ndct))
                    if np.sum(p_ndct < args.data_range*20/255) < air_num:
                        ldct_patches.append(p_ldct)
                        ndct_patches.append(p_ndct)
            if self.print_progress:
                print('image:' + str(n + 1) + '/' + str(len(ldct_images)) + ' is finished ^_^')
        ldct_patches = torch.from_numpy(np.array(ldct_patches)).unsqueeze(dim=1)
        ndct_patches = torch.from_numpy(np.array(ndct_patches)).unsqueeze(dim=1)
        return ldct_patches, ndct_patches


def show(img, title=None, cbar=False, figsize=None):
    if type(img) == torch.Tensor:
        img = np.array(img[0])
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

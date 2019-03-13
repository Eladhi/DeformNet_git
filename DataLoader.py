import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path
import scipy.io as sio
from scipy import ndimage
import numpy as np

# the dataloader collects the data from the following directories:
# + / (script folder)
#	+ jp2
#		+ training_data
#			+ img_folder
#					+ input.png
#					+ y.png
#					+ ux.mat
#					+ uy.mat
#		+ test_data
#			+ img_folder
#					+ input.png
#					+ y.png
#					+ ux.mat
#					+ uy.mat
#
# supporting other formats may be acheived easily by fitting the format's dir tree in the init function
class DA_jp2000_dataloader(Dataset):
    def __init__(self, R=100, wanted_img_size=[50, 50], Train=True, transform=None, use_cuda=False):
        self.transform = transform
        self.wanted_img_size = wanted_img_size
        self.use_cuda = use_cuda
        self.Train = Train
        # set data folder path
        if Train:
            path = 'jp2/training_data/'
            #path = 'Affine2/training_data/'
            #path = 'StaticTransform/training_data/'
        else:
            path = 'jp2/test_data/'
            #path = 'Affine2/test_data/'
            #path = 'StaticTransform/test_data/'
        # create self.content containing input, ux, uy path
        img_folders_list = [file for file in os.listdir(path) if os.path.isdir(path + file)]
        self.content = {}
        self.content['input'] = []
        self.content['ux'] = []
        self.content['uy'] = []
        self.content['deformed'] = []
        for img_folder in img_folders_list:
            tmp_img_path = path + img_folder + '/' + 'R=' + str(R) + '/'
            #tmp_img_path = path + img_folder + '/'
            # make sure image is larger than given dimensions
            img = ndimage.imread(tmp_img_path + 'Input.png', mode="RGB")
            if img.shape[0] >= wanted_img_size[0] and img.shape[1] >= wanted_img_size[1]:
                self.content['input'] += [tmp_img_path + 'Input.png']
                self.content['ux'] += [tmp_img_path + 'ux.mat']
                self.content['uy'] += [tmp_img_path + 'uy.mat']
                self.content['deformed'] += [tmp_img_path + 'y.png']
        self.img_folders_num = len(self.content['input'])

    def __len__(self):
        return self.img_folders_num

    def __getitem__(self, idx):
        #print(self.content['input'][idx])
        img = ndimage.imread(self.content['input'][idx], mode="RGB")
        ux_mat = sio.loadmat(self.content['ux'][idx])['ux']
        uy_mat = sio.loadmat(self.content['uy'][idx])['uy']
        deformed = ndimage.imread(self.content['deformed'][idx], mode="RGB")

        # crop to wanted image size
        actual_h, actual_w = img.shape[0], img.shape[1]
        wanted_h, wanted_w = self.wanted_img_size[0], self.wanted_img_size[1]
        if self.Train:  # random crop when train
            h_rand = int(np.random.randint(low=0, high=actual_h-wanted_h+1)/64)*64
            w_rand = int(np.random.randint(low=0, high=actual_w-wanted_w+1)/64)*64
            img = img[h_rand:h_rand+wanted_h, w_rand:w_rand+wanted_w,:]
            ux_mat = ux_mat[h_rand:h_rand + wanted_h, w_rand:w_rand + wanted_w]
            uy_mat = uy_mat[h_rand:h_rand + wanted_h, w_rand:w_rand + wanted_w]
            #print("%d-%d, %d, %d-%d, %d" % (h_rand, h_rand + wanted_h, actual_h, w_rand, w_rand + wanted_w, actual_w))
            deformed = deformed[h_rand:h_rand + wanted_h, w_rand:w_rand + wanted_w, :]
        else:
            #h_idx = int((int(actual_h / 2) - int(wanted_h / 2))/64)*64
            h_idx = int(((actual_h / 2) - (wanted_h / 2)) / 64) * 64
            #w_idx = int((int(actual_w / 2) - int(wanted_w / 2))/64)*64
            w_idx = int(((actual_w / 2) - (wanted_w / 2)) / 64) * 64
            img = img[h_idx:h_idx+wanted_h, w_idx:w_idx+wanted_w,:]
            ux_mat = ux_mat[h_idx:h_idx + wanted_h, w_idx:w_idx + wanted_w]
            uy_mat = uy_mat[h_idx:h_idx + wanted_h, w_idx:w_idx + wanted_w]
            #print("%d-%d, %d, %d-%d, %d" % (h_idx, h_idx + wanted_h, actual_h, w_idx, w_idx + wanted_w, actual_w))
            deformed = deformed[h_idx:h_idx + wanted_h, w_idx:w_idx + wanted_w, :]

        if self.transform: # currently not good for random!!
            img = self.transform(img)
            ux_mat = self.transform(ux_mat)
            uy_mat = self.transform(uy_mat)
            deformed = self.transform(deformed)
        uxuy = [ux_mat, uy_mat]
        # make tensor
        if self.use_cuda:
            img = torch.cuda.FloatTensor(img).permute(2,0,1)
            uxuy = torch.cuda.FloatTensor(uxuy)
            deformed = torch.cuda.FloatTensor(deformed).permute(2,0,1)
        else:
            img = torch.FloatTensor(img).permute(2,0,1)
            uxuy = torch.FloatTensor(uxuy)
            deformed = torch.FloatTensor(deformed).permute(2,0,1)
        return img, uxuy, deformed

import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path
from scipy import ndimage

# the dataloader collects the data from the following directories:
# + / (script folder)
#	+ $dirpath
#			+ img_folder
#				+ 'R='# (( not necessarily, in a comment, for affine ))
#					+ Input.png
#
# supporting other formats may be acheived easily by fitting the format's dir tree in the init function
class DA_jp2000_ForwardDataloader(Dataset):
    def __init__(self, R=100, transform=None, use_cuda=False, dirpath=None):
        self.transform = transform
        self.use_cuda = use_cuda
        self.dirpath = dirpath
        # create self.content containing input, ux, uy path
        img_folders_list = [file for file in os.listdir(dirpath) if os.path.isdir(dirpath + file)]
        self.content = {}
        self.content['input'] = []
        self.content['deformed'] = []
        for img_folder in img_folders_list:
            tmp_img_path = dirpath + img_folder + '/' + 'R=' + str(R) + '/'
            #tmp_img_path = dirpath + img_folder + '/'
            # make sure image is larger than given dimensions
            #img = ndimage.imread(tmp_img_path + 'Input.png', mode="RGB")
            self.content['input'] += [tmp_img_path + 'Input.png']
            self.content['deformed'] += [tmp_img_path + 'y.png']
        self.img_folders_num = len(self.content['input'])

    def __len__(self):
        return self.img_folders_num

    def __getitem__(self, idx):
        img = ndimage.imread(self.content['input'][idx], mode="RGB")
        deformed = ndimage.imread(self.content['deformed'][idx], mode="RGB")
        if self.transform: # currently not good for random!!
            img = self.transform(img)
            deformed = self.transform(deformed)
        # make tensor
        if self.use_cuda:
            img = torch.cuda.FloatTensor(img).permute(2,0,1)
            deformed = torch.cuda.FloatTensor(deformed).permute(2, 0, 1)
        else:
            img = torch.FloatTensor(img).permute(2,0,1)
            deformed = torch.FloatTensor(deformed).permute(2, 0, 1)
        return img, deformed

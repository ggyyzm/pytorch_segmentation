# Originally written by Kazuto Nakashima
# https://github.com/kazuto1011/deeplab-pytorch

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
from dataloaders import XImage
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
import random


class SEN12MSDataset(BaseDataSet):
    def __init__(self, **kwargs):
        self.num_classes = 11
        self.palette = palette.get_sen12ms_palette(self.num_classes)
        super(SEN12MSDataset, self).__init__(**kwargs)

    def _set_files(self):
        # self.root = os.path.join(self.root, 'salinas')
        file_dir = os.listdir(self.root)
        train_dir = []
        val_dir = 'validation'
        self.files = []

        for season_dir in file_dir:
            if season_dir != 'validation':
                train_dir.append(season_dir)

        if self.val == False:
            for season_dir in train_dir:
                sub_train_dirs = os.listdir(os.path.join(self.root, season_dir))
                for sub_dir in sub_train_dirs:
                    if sub_dir[:2] == 's2':
                        self.files += os.listdir(os.path.join(self.root, season_dir, sub_dir))
            random.shuffle(self.files)
            # self.files = self.files[:16384]
        elif self.val == True:
            self.image_dir = os.path.join(self.root, val_dir, 's2_validation')
            self.label_dir = os.path.join(self.root, val_dir, 'lc_validation')
            self.files += os.listdir(self.image_dir)
        else:
            raise AttributeError("val parameter error")

    def label_simplified_IGBP(self, label):
        label = label[:, :, 0]

        result = (label==1)*1 + (label==2) * 1 +(label==3)*1 + (label==4)*1 + (label==5)*1 + (label==6)*2 + (label==7)*2\
                 + (label==8)*3 + (label==9)*3 + (label==10)*4 + (label==11)*5 + (label==12)*6 + (label==13)*7 +\
                 (label==14)*6 + (label==15)*8 + (label==16)*9 + (label==17)*10
        return result

    def _load_data(self, index):
        image_id = self.files[index]
        label_id = image_id.split("_", 3)[0]+'_'+image_id.split("_", 3)[1]+'_'+'lc'+'_'+image_id.split("_", 3)[3]
        if self.val == False:
            self.image_dir = os.path.join(self.root, image_id.split("_")[0]+'_'+image_id.split("_")[1], image_id.split("_")[2]+'_'+image_id.split("_")[3])
            self.label_dir = os.path.join(self.root, label_id.split("_")[0]+'_'+label_id.split("_")[1], label_id.split("_")[2]+'_'+label_id.split("_")[3])
        elif self.val == True:
            pass
        else:
            raise AttributeError("val parameter error")
        image_path = os.path.join(self.image_dir, image_id)
        label_path = os.path.join(self.label_dir, label_id)
        x_image = XImage.CXImage()
        x_image.Open(image_path)
        image = x_image.GetData(np.float32, data_arrange=0)
        x_label = XImage.CXImage()
        x_label.Open(label_path)
        label = x_label.GetData(np.int32, data_arrange=0)
        label = self.label_simplified_IGBP(label)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class SEN12MS(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):

        self.MEAN = [4867., 3486., 5786.]
        self.STD = [476., 654., 568.]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = SEN12MSDataset(**kwargs)  # 将kwargs中的键值对作为参数的键值对传入

        super(SEN12MS, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)    # -->BaseDataLoader


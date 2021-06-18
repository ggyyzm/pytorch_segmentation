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


class JANDataset(BaseDataSet):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, **kwargs):
        self.num_classes = 8
        self.palette = palette.get_voc_palette(self.num_classes)
        super(JANDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'jan')
        self.image_dir = os.path.join(self.root, '20200114')
        self.label_dir = os.path.join(self.root, '20200114gt')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.tiff')
        label_path = os.path.join(self.label_dir, image_id + '.tiff')
        x_image = XImage.CXImage()
        x_image.Open(image_path)
        image = x_image.GetData(np.float32, data_arrange=0)
        x_label = XImage.CXImage()
        x_label.Open(label_path)
        label = x_label.GetData(np.int32, data_arrange=0)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class JAN(BaseDataLoader):
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

        self.dataset = JANDataset(**kwargs)  # 将kwargs中的键值对作为参数的键值对传入

        super(JAN, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)    # -->BaseDataLoader


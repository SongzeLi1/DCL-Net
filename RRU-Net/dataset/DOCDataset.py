import os
import cv2
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset,DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.transform import train_transform
from paddleocr import PaddleOCR, draw_ocr, draw_ocr_box_txt
from PIL import Image, ImageDraw, ImageFont


class DOCDataset(Dataset):
    def __init__(self, names, imgs_dir, masks_dir, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.names = names
        logging.info(f'Creating dataset with {len(self.names)} examples')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = Image.open(self.imgs_dir + name).convert('RGB')
        if self.masks_dir is not None:
            mask_name = name.replace('psc_', 'gt3_')
            mask_name = mask_name.replace('mosaic_', 'c_')
            mask_name = mask_name.replace('.jpg', '.png')
            mask_name = mask_name.replace('.JPG', '.png')
            mask_name = mask_name.replace('.tif', '.png')
            mask_name = mask_name.replace('pngps', 'pngms')
            # mask_name = mask_name.replace('_qf70.jpg', '.png')
            mask = Image.open(self.masks_dir + mask_name).convert('L')
        else:
            mask = Image.open(self.imgs_dir + name).convert('RGB').convert('L')
        # image = cv2.imread(img_file[0], cv2.IMREAD_COLOR) # 不含alpha通道
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED) # 含alpha通道
        # mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        image = np.array(image, np.uint8)
        mask = np.array(mask)
        # print('mask:', mask.min(), mask.max())
        # print(image.shape, ocrlabel.shape, mask.shape)

        # 二分类
        # if mask.max() > 1: mask = np.uint8(mask / 255)
        # ---三分类 docimg---
        mask[mask == 255] = 0
        mask[mask == 76] = 1
        mask[mask == 29] = 2
        # # ---二分类 docimg Tamper和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 1
        # # ---二分类 docimg Orig和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 0
        # # ---二分类 Alinew, SUPATLANTIQUE, findit---
        # mask[mask !=0] = 1

        # print(np.array(image).min(), np.array(image).max()) # 0, 255
        # print(np.array(mask).min(), np.array(mask).max()) # 0, 1

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': mask
        }

        # if sum(map(sum, mask)) != 0:
        #     if self.transform:
        #         transformed = self.transform(image=image, mask=mask)
        #         image = transformed['image']
        #         mask = transformed['mask']
        #     return {
        #         'image': image,
        #         'label': mask
        #     }
        # else:
        #     pass


# class DOCDataset(Dataset):
#     def __init__(self, names, imgs_dir, masks_dir, transform):
#         self.transform = transform
#         self.names = names
#         logging.info(f'Creating dataset with {len(self.names)} examples')
#
#     def __len__(self):
#         return len(self.names)
#
#     def __getitem__(self, i):
#         name = self.names[i]
#         image = Image.open(name).convert('RGB')
#         mask_name = name.replace('_images_', '_gt3_')
#         mask_name = mask_name.replace('psc_', 'gt3_')
#         mask = Image.open(mask_name).convert('L')
#         image = np.array(image, np.uint8)
#         mask = np.array(mask)
#         # ---二分类 docimg Orig和Mosaic一类---
#         mask[mask == 255] = 0
#         mask[mask == 76] = 1
#         mask[mask == 29] = 0
#         # # ---二分类 Alinew, SUPATLANTIQUE---
#         # mask[mask !=0] = 1
#         if self.transform:
#             transformed = self.transform(image=image, mask=mask)
#             image = transformed['image']
#             mask = transformed['mask']
#         return {
#             'image': image,
#             'label': mask
#         }


if __name__ == '__main__':
    data_dir = "/pubdata/zhengkengtao/docimg/docimg_split811/train_images/"
    labels_dir = "/pubdata/zhengkengtao/docimg/docimg_split811/train_gt3/"
    # data_dir = "/pubdata/zhengkengtao/docimg/docimg_split811/crop224x224/train_images/"
    # labels_dir =  "/pubdata/zhengkengtao/docimg/docimg_split811/crop224x224/train_gt3/"
    # data_dir = "/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split811/train_imgs/"
    # labels_dir = "/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split811/train_gt/"
    img_names = os.listdir(data_dir)
    # for img in files[:100]:
    #     img_names.append(img.split('.')[0])
    #     img_names.sort()
    img_names.sort()
    print(len(img_names))
    train_data = DOCDataset(img_names, data_dir, labels_dir, train_transform([224, 224]))
    train_loader = DataLoader(dataset=train_data, batch_size=6, shuffle=False)
    for batch_idx, batch_samples in enumerate(train_loader):
        image, target = batch_samples['image'], batch_samples['label']
        print(image.shape, image.min(), image.max()) # -x, y
        print(target.shape, target.min(), target.max()) # 0, 2







import os
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn as nn
import numpy as np
import cv2
import random
import torch
import torch.nn.functional as F
from PIL import Image
def SRM(imgs):
    # SQUARE 5×5
    filter1 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    # SQUARE 3×3   
    filter2 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    ## Vertical second-order
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, -2, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    ## Horizontal second-order
    filter4 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    ## Horizontal first-order
    filter5 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, -1, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    ## Vertical first-order
    filter6 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, -1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    
    filter1 = np.asarray(filter1, dtype=float) / 12
    filter2 = np.asarray(filter2, dtype=float) / 4
    filter3 = np.asarray(filter3, dtype=float) / 2
    filter4 = np.asarray(filter4, dtype=float) / 2
    filter5 = np.asarray(filter5, dtype=float)
    filter6 = np.asarray(filter6, dtype=float)

    filters = []
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3],
    [filter4, filter4, filter4], [filter5, filter5, filter5], [filter6, filter6, filter6]]  # (3,3,5,5)
    filters = torch.FloatTensor(filters)    # (3,3,5,5)
    imgs = np.array(imgs, dtype=float)  # (375,500,3)
    imgs = np.einsum('klij->kjli', imgs)
    input = torch.tensor(imgs, dtype=torch.float32)

    op1 = F.conv2d(input, filters, stride=1, padding=2)
    op1 = op1[0]
    op1 = np.round(op1)
    op1[op1 > 2] = 2
    op1[op1 < -2] = -2
    return op1
    
class DataSetLoader(Dataset):
    def __init__(self, dataList, labelList):
        super(DataSetLoader, self).__init__()
        self.dataList = dataList
        self.labelList = labelList
    def __getitem__(self, index):
        try:
            image = Image.open(self.dataList[index]).convert('RGB')
            label = np.array(self.labelList[index])
            imageArray = np.asarray(image)
            srm = SRM([imageArray])
            srm = np.einsum('jkl->klj', srm).astype(np.uint8)


            image = ToTensor()(image)
            label = torch.LongTensor(label)
            srm = ToTensor()(srm)
        except UserWarning:
            print(self.dataList[index])
            return None
        else:
            return image, srm, label
    def __len__(self):
        return len(self.dataList)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight,nonlinearity='leaky_relu')
        m.bias.data.fill_(0.00)
        
def read_list(path, mode, subsample = 1):
    pathlist = []
    if mode == "tamper":
        files = os.listdir(path)
        files.sort()
        # files = files[:2000]
        for file in files:
            pathlist.append(path + file)
    if mode == "real":
        files = os.listdir(path)
        random_list = random.sample(range(0, len(files)), subsample)
        for i in random_list:
            pathlist.append(path + files[i])
    return pathlist

def read_list_realall(path):
    pathlist = []
    files = os.listdir(path)
    files.sort()
    # files = files[:2000]
    for file in files:
        pathlist.append(path + file)
    return pathlist

if __name__ == '__main__':
    train_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/train_tamper/',
                                  mode="tamper")
    lenght = len(train_list_tamper)
    train_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/train_real/', mode="real",
                                subsample=lenght)
    print(lenght)
    val_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/val_tamper/',
                                mode="tamper")
    lenght = len(val_list_tamper)
    val_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/val_real/', mode="real",
                              subsample=lenght)
    print(lenght)
    test_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/test_tamper/',
                                 mode="tamper")
    lenght = len(test_list_tamper)
    test_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/test_real/', mode="real",
                               subsample=lenght)
    print(lenght)
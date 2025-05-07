import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.nn as nn
import torch
import torch.nn.functional as F
from numpy.lib.function_base import corrcoef
from models import *
import argparse
from time import time
import pandas as pd
from random import shuffle
import random
from tqdm import tqdm
import cv2
from PIL import Image
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import warnings
from PIL import ImageFile
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
    def __init__(self, dataList):
        super(DataSetLoader, self).__init__()
        self.dataList = dataList
    def __getitem__(self, index):
        image = Image.open(self.dataList[index]).convert('RGB')
        image_name = self.dataList[index].split("/", -1)[-1]
        imageArray = np.asarray(image)
        srm = SRM([imageArray])
        srm = np.einsum('jkl->klj', srm).astype(np.uint8)


        image = ToTensor()(image)
        srm = ToTensor()(srm)
        return image, image_name, srm
    def __len__(self):
        return len(self.dataList)
def read_list(path):
    pathlist = []
    files = os.listdir(path)
    for file in files:
        pathlist.append(os.path.join(path, file))
    return pathlist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1) # 只能设置为1，否则出错
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--step_size', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.9)
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86_map/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86/")
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/mapblock/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/block/")
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/mapblock/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/block/")
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/mapblock/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/block/")
    parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/testdocimgsplit811_testtampermosaicorig400_crop512stride256/mapblock/")
    parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/testdocimgsplit811_testtampermosaicorig400_crop512stride256/block/")
    return parser.parse_args()
def test(Test_loader, model, cfg):
    correct = 0
    total = 0
    if not os.path.exists(cfg.outmap_dir): os.makedirs(cfg.outmap_dir)
    if not os.path.exists(cfg.out_dir): os.makedirs(cfg.out_dir)
    model.eval()
    with torch.no_grad():
        for data in tqdm(Test_loader):
            image, image_name, srm = data
            image, srm = image.to(cfg.device), srm.to(cfg.device)
            _, _, height, width = np.shape(image)
            output = model(image, srm) # shape: (b, 2)
            _, predict = torch.max(output.data, dim = 1) 
            predict_label = predict[0]
            if predict_label == 0:
                predict_mask = np.zeros((height, width))
            else:
                predict_mask = np.ones((height, width))
            # # 7.30 zkt：增加了概率图保存（为了算AUC），原代码没有
            output = output.cpu().data.numpy()
            map = np.ones((height, width)) * output[:, 1]
            cv2.imwrite(os.path.join(cfg.outmap_dir + image_name[0]), np.round(map * 255.0))
            cv2.imwrite(os.path.join(cfg.out_dir + image_name[0]), np.round(predict_mask * 255.0))
    # print('accuracy on Test set: %f %%' % float(100 * correct / total))
def main(cfg):
    # test_list_tamper = read_list('/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/select/test_tamper/')
    # test_list_real = read_list('/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/select/test_real/')
    # print(len(test_list_tamper))
    # print(len(test_list_real))
    # test_list = test_list_tamper + test_list_real
    # test_list = read_list('/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/img/')

    # test_list = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/test_images/')

    # test_list = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/train_images_1/') + \
    #             read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/train_images_2/') + \
    #             read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/val_images/') + \
    #             read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/test_images/')

    test_list = read_list('/data1/zhengkengtao/docimg/docimg_split811/testorig_crop512stride256/') + \
                read_list('/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride256/')

    test_list.sort()
    # test_list = test_list[0*200000:1*200000]
    print(len(test_list))
    # shuffle(test_list)
    test_set = DataSetLoader(test_list)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=cfg.batch_size, shuffle=False)

    # modelTest = DIDNet().to(cfg.device)
    # modelTest = nn.DataParallel(modelTest, device_ids = [0,1])
    modelTest = DIDNet()
    modelTest = nn.DataParallel(modelTest)
    pretrained_dict = torch.load('/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/DIDNet_epoch_86.pth')
    # pretrained_dict = torch.load('/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/DIDNet_epoch_73.pth')

    modelTest.load_state_dict(pretrained_dict['state_dict'])
    test(test_loader, modelTest, cfg)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

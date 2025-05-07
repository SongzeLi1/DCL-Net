import os
import random
# from utils.load import *
from utils.utils import *
import time
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm


# 2022.4.2
# # ------------对所有文档图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/DocImg/PS/tamper_mosaic/'
# val_percent = 0.05
# imgs_name = os.listdir(img_path)
# print(imgs_name)
# coms, spls, adds, rems, rmas = [], [], [], [], []
# for img_name in imgs_name:
#     if '_com_' in img_name:
#         coms.append(img_name)
#     elif '_spl_' in img_name:
#         spls.append(img_name)
#     elif '_add_' in img_name:
#         adds.append(img_name)
#     elif '_rem_' in img_name:
#         rems.append(img_name)
#     elif '_rma_' in img_name:
#         rmas.append(img_name)
#     else:
#         print(img_name)
# print('com:', len(coms))
# print('spl:', len(spls))
# print('add:', len(adds))
# print('rem:', len(rems))
# print('rma:', len(rmas))
# train_f = open('/pubdata/zhengkengtao/DocImg/PS/tamper_mosaic_train[{}].txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/DocImg/PS/tamper_mosaic_val[{}].txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# com_train = coms[:round(len(coms)*train_percent)]
# spl_train = spls[:round(len(spls)*train_percent)]
# add_train = adds[:round(len(adds)*train_percent)]
# rem_train = rems[:round(len(rems)*train_percent)]
# rma_train = rmas[:round(len(rmas)*train_percent)]
# print('com_train:', len(com_train))
# print('spl_train:', len(spl_train))
# print('add_train:', len(add_train))
# print('rem_train:', len(rem_train))
# print('rma_train:', len(rma_train))
# train = com_train + spl_train + add_train + rem_train + rma_train
# # val
# val = imgs_name
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('_ps_', '_ms_')
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('_ps_', '_ms_')
#     print(val_name + ',' + val_mask_name, file=val_f)


# # 2022.4.11
# # ------------对所有文档图像划分训练集和验证集，按名称排序后后面的部分作为验证集---------------
# img_path = '/pubdata/zhengkengtao/DocImg/PS/tamper_mosaic/'
# val_percent = 0.05
# imgs_name = os.listdir(img_path)
# imgs_name.sort()
# print(imgs_name)
# train_f = open('/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_train[{}].txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_val[{}].txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# train = imgs_name[:round(len(imgs_name)*train_percent)]
# # val
# val = imgs_name
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('_ps_', '_ms_')
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('_ps_', '_ms_')
#     print(val_name + ',' + val_mask_name, file=val_f)


# 2022.4.11
# # ----------对文档图像crop后的图像块选择一定篡改区域比例的图像-----------（运行速度很慢）
# d = 512 # 768
# img_path = '/pubdata/zhengkengtao/DocImg/PS/crop{}/tamper_mosaic_crop_{}/'.format(d, d)
# mask_path = '/pubdata/zhengkengtao/DocImg/PS/crop{}/mask_mosaic_crop_{}/'.format(d, d)
# imgs_name = os.listdir(img_path)
# masks_name = os.listdir(mask_path)
# rate_f = open('/pubdata/zhengkengtao/DocImg/PS/crop{}/tamper_mosaic_crop_{}_tr.txt'.format(d, d), 'w+')
# all = d*d
# selected_tampers = []
# selected_masks = []
# for mask_name in tqdm(masks_name):
#     print(mask_name)
#     mask = Image.open(mask_path + mask_name)
#     mask = mask.convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     # print(mask.min(), mask.max())
#     mask_norm = np.array(mask / 255, dtype=np.uint8)
#     sum_all = sum(sum(i) for i in mask_norm)
#     print(mask_name + ' {:.4f}'.format(sum_all / all))
#     print(mask_name + ' {:.4f}'.format(sum_all / all), file=rate_f)
# # 2022.4.11
# d = 512 # 768
# tr_txt = '/pubdata/zhengkengtao/DocImg/PS/crop{}/tamper_mosaic_crop_{}_tr.txt'.format(d, d)
# masks = []
# # low_pixel = int(all * 0.1)
# # high_pixel = int(all * 0.5)
# # print(low_pixel, high_pixel)
# use_f = open('/pubdata/zhengkengtao/DocImg/PS/crop{}/tamper_mosaic_crop_{}_[trover0].txt'.format(d, d), 'w+')
# with open(tr_txt, 'r') as f1:
#     for line in f1:
#         mask_name = line.strip('\n').split(' ')[0]
#         tr = line.strip('\n').split(' ')[1]
#         if float(tr) > 0:
#             tamper_name = mask_name.replace('_ms_', '_ps_')
#             print(tamper_name + ',' + mask_name, file=use_f)

# # 2022.4.11
# # ---------对文档图像训练集图像进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# train_txt = '/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_train[0.95].txt'
# crop_tamperselect_txt = '/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_crop_512_[trover0].txt'
# train_cropselect_txt = '/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_train[0.95]_crop_512_[trover0].txt'
# train_imgs = []
# crop_tamperselect_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(train_txt, 'r') as f1:
#     for line in f1:
#         train_img = line.strip('\n').split(',')[0]
#         train_imgs.append(train_img)
# with open(crop_tamperselect_txt, 'r') as f2:
#     for line in f2:
#         crop_tamperselect_img = line.strip('\n').split(',')[0]
#         crop_tamperselect_imgs.append(crop_tamperselect_img)
# ftrain = open(train_cropselect_txt, 'w+')
# for train_img in train_imgs:
#     for crop_tamperselect_img in crop_tamperselect_imgs:
#         if train_img[:-4] in crop_tamperselect_img:
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('_ps_', '_ms_'))
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('_ps_', '_ms_'), file=ftrain)
# # ---------对验证集图像分类型进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# val_txt = '/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_val[0.05].txt'
# crop_tamperselect_txt = '/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_crop_512_[trover0].txt'
# val_cropselect_txt = '/pubdata/zhengkengtao/DocImg/PS/crop512/tamper_mosaic_val[0.05]_crop_512_[trover0].txt'
# val_imgs = []
# crop_tamperselect_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(val_txt, 'r') as f1:
#     for line in f1:
#         val_img = line.strip('\n').split(',')[0]
#         val_imgs.append(val_img)
# with open(crop_tamperselect_txt, 'r') as f2:
#     for line in f2:
#         crop_tamperselect_img = line.strip('\n').split(',')[0]
#         crop_tamperselect_imgs.append(crop_tamperselect_img)
# fval = open(val_cropselect_txt, 'w+')
# for val_img in val_imgs:
#     for crop_tamperselect_img in crop_tamperselect_imgs:
#         if val_img[:-4] in crop_tamperselect_img:
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('_ps_', '_ms_'))
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('_ps_', '_ms_'), file=fval)

# # 统计各篡改类型图像块数量
# img_path = '/pubdata/zhengkengtao/DocImg/PS/crop/tamper_mosaic_train[0.95]_crop_768_[trover0]/'
# img_names = os.listdir(img_path)
# coms, spls, adds, rems, rmas = [], [], [], [], []
# for img_name in img_names:
#     if '_com_c' in img_name:
#         coms.append(img_name)
#     elif '_spl_c' in img_name:
#         spls.append(img_name)
#     elif '_add_c' in img_name:
#         adds.append(img_name)
#     elif '_rem_c' in img_name:
#         rems.append(img_name)
#     elif 'rma_c' in img_name:
#         rmas.append(img_name)
# print(len(coms), len(spls), len(adds), len(rems), len(rmas))

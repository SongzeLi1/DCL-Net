import os
import random
# from utils.load import *
from utils.utils import *
import shutil
import time
from PIL import Image
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm


# # ------------对所有证书图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/certificate/mask_png/'
# val_percent = 0.1
# imgs_name = os.listdir(img_path)
# coms, spls, adds, rems, rmas = [], [], [], [], []
# for img_name in imgs_name:
#     if '_com' in img_name:
#         coms.append(img_name)
#     elif '_spl' in img_name:
#         spls.append(img_name)
#     elif '_add' in img_name:
#         adds.append(img_name)
#     elif '_rem' in img_name:
#         rems.append(img_name)
#     elif '_rma' in img_name:
#         rmas.append(img_name)
#     else:
#         print(img_name)
# print('com:', len(coms))
# print('spl:', len(spls))
# print('add:', len(adds))
# print('rem:', len(rems))
# print('rma:', len(rmas))
# train_f = open('/pubdata/zhengkengtao/certificate/train[{}].txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/certificate/val[{}].txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# com_train = coms[:round(len(coms)*train_percent)]
# spl_train = spls[:round(len(spls)*train_percent)]
# add_train = adds[:round(len(adds)*train_percent)]
# rem_train = rems[:round(len(rems)*train_percent)]
# rma_train = rmas[:round(len(rmas)*train_percent)]
# print('com_train:', len(com_train))
# print('spl_train:', len(spl_train))  # 898,之前895
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
#     train_mask_name = train_name.replace('ps', 'ms', 1)
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('ps', 'ms', 1)
#     print(val_name + ',' + val_mask_name, file=val_f)


# ----------对crop后的图像块选择一定篡改区域比例的图像，再对挑出的图像划分训练验证集，train和val对于每种篡改类型比例一致-----------（运行速度很慢）
# # 1.挑选图像块
# img_path = '/pubdata/zhengkengtao/certificate/tamper_crop_512/'
# mask_path = '/pubdata/zhengkengtao/certificate/mask_crop_512/'
# imgs_name = os.listdir(img_path)
# masks_name = os.listdir(mask_path)
# use_f = open('/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5].txt', 'w+')
# all = 512*512
# low_pixel = int(all * 0.1)
# high_pixel = int(all * 0.5)
# print(low_pixel, high_pixel)
# selected_tampers = []
# selected_masks = []
# for mask_name in masks_name:
#     print(mask_name)
#     mask = Image.open(mask_path + mask_name)
#     mask = mask.convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     # print(mask.min(), mask.max())
#     mask_norm = np.array(mask / 255, dtype=np.uint8)
#     sum_all = sum(sum(i) for i in mask_norm)
#     if sum_all>=low_pixel and sum_all<=high_pixel:
#         selected_masks.append(mask_name)
#         tmp = mask_name
#         tamper_name = tmp.replace('pngms', 'pngps', 1)
#         selected_tampers.append(tamper_name)
#         print(tamper_name + ',' + mask_name, file=use_f)
# # 2.对挑选出的图像块分train和val,train和val对于每种篡改类型比例一致
# file_path = '/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5].txt'
# selected_tampers = []
# with open(file_path, 'r') as f:
#     for line in f:
#         img = line.strip('\n').split(',')[0]
#         selected_tampers.append(img)
# val_percent = 0.1
# train_percent = 1 - val_percent
# train_f = open('/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5]_train[{}].txt'.format(train_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5]_val[{}].txt'.format(val_percent), 'w+')
# coms, spls, adds, rems, rmas = [], [], [], [], []
# for tamper in selected_tampers:
#     if '_com' in tamper:
#         coms.append(tamper)
#     elif '_spl' in tamper:
#         spls.append(tamper)
#     elif '_add' in tamper:
#         adds.append(tamper)
#     elif '_rem' in tamper:
#         rems.append(tamper)
#     elif '_rma' in tamper:
#         rmas.append(tamper)
# # train
# com_train = coms[:round(len(coms)*train_percent)]
# spl_train = spls[:round(len(spls)*train_percent)]
# add_train = adds[:round(len(adds)*train_percent)]
# rem_train = rems[:round(len(rems)*train_percent)]
# rma_train = rmas[:round(len(rmas)*train_percent)]
# train = com_train + spl_train + add_train + rem_train + rma_train
# # val
# val = selected_tampers
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('pngps', 'pngms', 1)
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('pngps', 'pngms', 1)
#     print(val_name + ',' + val_mask_name, file=val_f)


# # ---------对训练集图像进行crop，再选择一定篡改区域比例的那些图像-------------------------------
# train_txt = '/pubdata/zhengkengtao/certificate/train[0.9].txt'
# crop_tamperselect_txt = '/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5].txt'
# train_crop_txt = '/pubdata/zhengkengtao/certificate/train[0.9]_crop_512_[tamper0.1-0.5].txt'
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
# f = open(train_crop_txt, 'w+')
# for train_img in train_imgs:
#     for crop_tamperselect_img in crop_tamperselect_imgs:
#         if train_img[:-4] in crop_tamperselect_img:
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1))
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1), file=f)
# # ---------对验证集图像进行crop，再选择一定篡改区域比例的那些图像------------------------------
# val_txt = '/pubdata/zhengkengtao/certificate/val[0.1].txt'
# crop_tamperselect_txt = '/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5].txt'
# val_crop_txt = '/pubdata/zhengkengtao/certificate/val[0.1]_crop_512_[tamper0.1-0.5].txt'
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
# f = open(val_crop_txt, 'w+')
# for val_img in val_imgs:
#     for crop_tamperselect_img in crop_tamperselect_imgs:
#         if val_img[:-4] in crop_tamperselect_img:
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1))
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1), file=f)


# # ---------对训练集图像进行crop-------------------------------
# train_txt = '/pubdata/zhengkengtao/certificate/train[0.9].txt'
# crop_txt = '/pubdata/zhengkengtao/certificate/crop_512.txt'
# train_crop_txt = '/pubdata/zhengkengtao/certificate/train[0.9]_crop_512.txt'
# train_imgs = []
# crop_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(train_txt, 'r') as f1:
#     for line in f1:
#         train_img = line.strip('\n').split(',')[0]
#         train_imgs.append(train_img)
# with open(crop_txt, 'r') as f2:
#     for line in f2:
#         crop_img = line.strip('\n').split(',')[0]
#         crop_imgs.append(crop_img)
# f_train = open(train_crop_txt, 'w+')
# for train_img in train_imgs:
#     for crop_img in crop_imgs:
#         if train_img[:-4] in crop_img:
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1))
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1), file=f_train)
# # ---------对验证集图像进行crop-------------------------------
# val_txt = '/pubdata/zhengkengtao/certificate/val[0.1].txt'
# crop_txt = '/pubdata/zhengkengtao/certificate/crop_512.txt'
# val_crop_txt = '/pubdata/zhengkengtao/certificate/val[0.1]_crop_512.txt'
# val_imgs = []
# crop_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(val_txt, 'r') as f1:
#     for line in f1:
#         val_img = line.strip('\n').split(',')[0]
#         val_imgs.append(val_img)
# with open(crop_txt, 'r') as f2:
#     for line in f2:
#         crop_img = line.strip('\n').split(',')[0]
#         crop_imgs.append(crop_img)
# f_val = open(val_crop_txt, 'w+')
# for val_img in val_imgs:
#     for crop_img in crop_imgs:
#         if val_img[:-4] in crop_img:
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1))
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1), file=f_val)


# # ------------对所有证书图像按篡改类型分类---------------
# img_path = '/pubdata/zhengkengtao/certificate/tamper_png/'
# val_percent = 0.1
# imgs_name = os.listdir(img_path)
# coms, spls, adds, rems, rmas = [], [], [], [], []
# com_f = open('/pubdata/zhengkengtao/certificate/com.txt', 'w+')
# spl_f = open('/pubdata/zhengkengtao/certificate/spl.txt', 'w+')
# add_f = open('/pubdata/zhengkengtao/certificate/add.txt', 'w+')
# rem_f = open('/pubdata/zhengkengtao/certificate/rem.txt', 'w+')
# rma_f = open('/pubdata/zhengkengtao/certificate/rma.txt', 'w+')
# for img_name in imgs_name:
#     if '_com' in img_name:
#         coms.append(img_name)
#         print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=com_f)
#     elif '_spl' in img_name:
#         spls.append(img_name)
#         print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=spl_f)
#     elif '_add' in img_name:
#         adds.append(img_name)
#         print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=add_f)
#     elif '_rem' in img_name:
#         rems.append(img_name)
#         print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=rem_f)
#     elif '_rma' in img_name:
#         rmas.append(img_name)
#         print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=rma_f)
#     else:
#         print(img_name)
# print('com:', len(coms))
# print('spl:', len(spls))
# print('add:', len(adds))
# print('rem:', len(rems))
# print('rma:', len(rmas))


# train_txt = '/pubdata/zhengkengtao/certificate/train[0.9].txt'
# train_imgs = []
# com_train_f = open('/pubdata/zhengkengtao/certificate/com[0.9].txt', 'w+')
# spl_train_f = open('/pubdata/zhengkengtao/certificate/spl[0.9].txt', 'w+')
# add_train_f = open('/pubdata/zhengkengtao/certificate/add[0.9].txt', 'w+')
# rem_train_f = open('/pubdata/zhengkengtao/certificate/rem[0.9].txt', 'w+')
# rma_train_f = open('/pubdata/zhengkengtao/certificate/rma[0.9].txt', 'w+')
# with open(train_txt, 'r') as f1:
#     for line in f1:
#         img_name = line.strip('\n').split(',')[0]
#         if '_com' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=com_train_f)
#         elif '_spl' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=spl_train_f)
#         elif '_add' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=add_train_f)
#         elif '_rem' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=rem_train_f)
#         elif '_rma' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=rma_train_f)
#         else:
#             print(img_name)


# train_txt = '/pubdata/zhengkengtao/certificate/val[0.1].txt'
# train_imgs = []
# com_train_f = open('/pubdata/zhengkengtao/certificate/com[0.1].txt', 'w+')
# spl_train_f = open('/pubdata/zhengkengtao/certificate/spl[0.1].txt', 'w+')
# add_train_f = open('/pubdata/zhengkengtao/certificate/add[0.1].txt', 'w+')
# rem_train_f = open('/pubdata/zhengkengtao/certificate/rem[0.1].txt', 'w+')
# rma_train_f = open('/pubdata/zhengkengtao/certificate/rma[0.1].txt', 'w+')
# with open(train_txt, 'r') as f1:
#     for line in f1:
#         img_name = line.strip('\n').split(',')[0]
#         if '_com' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=com_train_f)
#         elif '_spl' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=spl_train_f)
#         elif '_add' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=add_train_f)
#         elif '_rem' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=rem_train_f)
#         elif '_rma' in img_name:
#             print(img_name + ',' + img_name.replace('pngps', 'pngms', 1), file=rma_train_f)
#         else:
#             print(img_name)


# # ---------对训练集图像分类型进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# # 'com', 'spl', 'add', 'rem', 'rma'
# tamper_type = 'rma'
# train_txt = '/pubdata/zhengkengtao/certificate/{}[0.9].txt'.format(tamper_type)
# crop_tamperselect_txt = '/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5].txt'
# train_crop_txt = '/pubdata/zhengkengtao/certificate/{}[0.9]_crop_512_[tamper0.1-0.5].txt'.format(tamper_type)
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
# f = open(train_crop_txt, 'w+')
# for train_img in train_imgs:
#     for crop_tamperselect_img in crop_tamperselect_imgs:
#         if train_img[:-4] in crop_tamperselect_img:
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1))
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1), file=f)
# # ---------对验证集图像分类型进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# val_txt = '/pubdata/zhengkengtao/certificate/{}[0.1].txt'.format(tamper_type)
# crop_tamperselect_txt = '/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5].txt'
# val_crop_txt = '/pubdata/zhengkengtao/certificate/{}[0.1]_crop_512_[tamper0.1-0.5].txt'.format(tamper_type)
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
# f = open(val_crop_txt, 'w+')
# for val_img in val_imgs:
#     for crop_tamperselect_img in crop_tamperselect_imgs:
#         if val_img[:-4] in crop_tamperselect_img:
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1))
#             print(crop_tamperselect_img + ',' + crop_tamperselect_img.replace('pngps', 'pngms', 1), file=f)


# # ---------对训练集图像分类型进行crop-------------------------------（运行速度较快）
# # 'com', 'spl', 'add', 'rem', 'rma'
# tamper_type = 'rma'
# train_txt = '/pubdata/zhengkengtao/certificate/{}[0.9].txt'.format(tamper_type)
# crop_txt = '/pubdata/zhengkengtao/certificate/crop_512.txt'
# train_crop_txt = '/pubdata/zhengkengtao/certificate/{}[0.9]_crop_512.txt'.format(tamper_type)
# train_imgs = []
# crop_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(train_txt, 'r') as f1:
#     for line in f1:
#         train_img = line.strip('\n').split(',')[0]
#         train_imgs.append(train_img)
# with open(crop_txt, 'r') as f2:
#     for line in f2:
#         crop_img = line.strip('\n').split(',')[0]
#         crop_imgs.append(crop_img)
# f = open(train_crop_txt, 'w+')
# for train_img in train_imgs:
#     for crop_img in crop_imgs:
#         if train_img[:-4] in crop_img:
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1))
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1), file=f)
# # ---------对验证集图像分类型进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# val_txt = '/pubdata/zhengkengtao/certificate/{}[0.1].txt'.format(tamper_type)
# crop_txt = '/pubdata/zhengkengtao/certificate/crop_512.txt'
# val_crop_txt = '/pubdata/zhengkengtao/certificate/{}[0.1]_crop_512.txt'.format(tamper_type)
# val_imgs = []
# crop_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(val_txt, 'r') as f1:
#     for line in f1:
#         val_img = line.strip('\n').split(',')[0]
#         val_imgs.append(val_img)
# with open(crop_txt, 'r') as f2:
#     for line in f2:
#         crop_img = line.strip('\n').split(',')[0]
#         crop_imgs.append(crop_img)
# f = open(val_crop_txt, 'w+')
# for val_img in val_imgs:
#     for crop_img in crop_imgs:
#         if val_img[:-4] in crop_img:
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1))
#             print(crop_img + ',' + crop_img.replace('pngps', 'pngms', 1), file=f)


# # ------------对所有SUPATLANTIQUE图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper/'
# val_percent = 0.1
# imgs_name = os.listdir(img_path)
# coms, spls, retouchs = [], [], []
# com_f = open('/pubdata/zhengkengtao/SUPATLANTIQUE/com.txt', 'w+')
# spl_f = open('/pubdata/zhengkengtao/SUPATLANTIQUE/spl.txt', 'w+')
# retouch_f = open('/pubdata/zhengkengtao/SUPATLANTIQUE/retouch.txt', 'w+')
# for img_name in imgs_name:
#     if 'com' in img_name:
#         coms.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=com_f)
#     elif 'spl' in img_name:
#         spls.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=spl_f)
#     elif 'retouch' in img_name:
#         retouchs.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=retouch_f)
#     else:
#         print(img_name)
# print('com:', len(coms))
# print('spl:', len(spls))
# print('retouch:', len(retouchs))
# train_f = open('/pubdata/zhengkengtao/SUPATLANTIQUE/train[{}].txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/SUPATLANTIQUE/val[{}].txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# com_train = coms[:round(len(coms)*train_percent)]
# spl_train = spls[:round(len(spls)*train_percent)]
# retouch_train = retouchs[:round(len(retouchs)*train_percent)]
# print('com_train:', len(com_train))
# print('spl_train:', len(spl_train))
# print('retouch_train:', len(retouch_train))
# train = com_train + spl_train + retouch_train
# # val
# val = imgs_name
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('.tif', '.png')
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('.tif', '.png')
#     print(val_name + ',' + val_mask_name, file=val_f)


# # ------------对所有Payslip图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/Payslip/tamper/'
# val_percent = 0.1
# imgs_name = os.listdir(img_path)
# cpinter_case1s, cpinter_case2s, cpinter_case3s, cpintras, imitation_case1s, imitation_case2s = [], [], [], [], [], []
# cpinter_case1_f = open('/pubdata/zhengkengtao/Payslip/cpinter_case1.txt', 'w+')
# cpinter_case2_f = open('/pubdata/zhengkengtao/Payslip/cpinter_case2.txt', 'w+')
# cpinter_case3_f = open('/pubdata/zhengkengtao/Payslip/cpinter_case3.txt', 'w+')
# cpintra_f = open('/pubdata/zhengkengtao/Payslip/cpintra.txt', 'w+')
# imitation_case1_f = open('/pubdata/zhengkengtao/Payslip/imitation_case1.txt', 'w+')
# imitation_case2_f = open('/pubdata/zhengkengtao/Payslip/imitation_case2.txt', 'w+')
# for img_name in imgs_name:
#     if 'cpinter_case1' in img_name:
#         cpinter_case1s.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=cpinter_case1_f)
#     elif 'cpinter_case2' in img_name:
#         cpinter_case2s.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=cpinter_case2_f)
#     elif 'cpinter_case3' in img_name:
#         cpinter_case3s.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=cpinter_case3_f)
#     elif 'cpintra' in img_name:
#         cpintras.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=cpintra_f)
#     elif 'imitation_case1' in img_name:
#         imitation_case1s.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=imitation_case1_f)
#     elif 'imitation_case2' in img_name:
#         imitation_case2s.append(img_name)
#         print(img_name + ',' + img_name.replace('.tif', '.png'), file=imitation_case2_f)
#     else:
#         print(img_name)
#
# print('cpinter_case1:', len(cpinter_case1s))
# print('cpinter_case2:', len(cpinter_case2s))
# print('cpinter_case3:', len(cpinter_case3s))
# print('cpintra:', len(cpintras))
# print('imitation_case1:', len(imitation_case1s))
# print('imitation_case2:', len(imitation_case2s))
# train_f = open('/pubdata/zhengkengtao/Payslip/train[{}].txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/Payslip/val[{}].txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# cpinter_case1_train = cpinter_case1s[:round(len(cpinter_case1s)*train_percent)]
# cpinter_case2_train = cpinter_case2s[:round(len(cpinter_case2s)*train_percent)]
# cpinter_case3_train = cpinter_case3s[:round(len(cpinter_case3s)*train_percent)]
# cpintra_train = cpintras[:round(len(cpintras)*train_percent)]
# imitation_case1_train = imitation_case1s[:round(len(imitation_case1s)*train_percent)]
# imitation_case2_train = imitation_case2s[:round(len(imitation_case2s)*train_percent)]
#
# print('cpinter_case1_train:', len(cpinter_case1_train))
# print('cpinter_case2_train:', len(cpinter_case2_train))
# print('cpinter_case3_train:', len(cpinter_case3_train))
# print('cpintra_train:', len(cpintra_train))
# print('imitation_case1_train:', len(imitation_case1_train))
# print('imitation_case2_train:', len(imitation_case2_train))
# train = cpinter_case1_train + cpinter_case2_train + cpinter_case3_train + cpintra_train + imitation_case1_train + imitation_case2_train
# # val
# val = imgs_name
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('.tif', '.png')
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('.tif', '.png')
#     print(val_name + ',' + val_mask_name, file=val_f)


# # ---------对SUPATLANTIQUE训练集图像进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# train_txt = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper为png格式的数据/train[0.9].txt'
# crop_train_txt = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper为png格式的数据/crop_512_[tamper0.1-0.5]_train[0.9].txt'
# train_crop_txt = '/pubdata/zhengkengtao/certificate/train[0.9]_crop_512_[tamper0.1-0.5].txt'
# train_imgs = []
# crop_train_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(train_txt, 'r') as f1:
#     for line in f1:
#         train_img = line.strip('\n').split(',')[0]
#         train_imgs.append(train_img)
# with open(crop_train_txt, 'r') as f2:
#     for line in f2:
#         crop_train_img = line.strip('\n').split(',')[0]
#         crop_train_imgs.append(crop_train_img)
# f = open(train_crop_txt, 'w+')
# for train_img in train_imgs:
#     for crop_train_img in crop_train_imgs:
#         if train_img[:-4] in crop_train_img:
#             print(crop_train_img + ',' + crop_train_img.replace('pngps', 'pngms', 1))
#             print(crop_train_img + ',' + crop_train_img.replace('pngps', 'pngms', 1), file=f)
# # ---------对验证集图像进行crop，再选择一定篡改区域比例的那些图像-------------------------------（运行速度较快）
# val_txt = '/pubdata/zhengkengtao/certificate/val[0.1].txt'
# crop_val_txt = '/pubdata/zhengkengtao/certificate/crop_512_[tamper0.1-0.5]_val[0.1].txt'
# val_crop_txt = '/pubdata/zhengkengtao/certificate/val[0.1]_crop_512_[tamper0.1-0.5].txt'
# val_imgs = []
# crop_val_imgs = []
# # 'r': 读取， 'a': 追加，'w'：覆盖
# with open(val_txt, 'r') as f1:
#     for line in f1:
#         val_img = line.strip('\n').split(',')[0]
#         val_imgs.append(val_img)
# with open(crop_val_txt, 'r') as f2:
#     for line in f2:
#         crop_val_img = line.strip('\n').split(',')[0]
#         crop_val_imgs.append(crop_val_img)
# f = open(val_crop_txt, 'w+')
# for val_img in val_imgs:
#     for crop_val_img in crop_val_imgs:
#         if val_img[:-4] in crop_val_img:
#             print(crop_val_img + ',' + crop_val_img.replace('pngps', 'pngms', 1))
#             print(crop_val_img + ',' + crop_val_img.replace('pngps', 'pngms', 1), file=f)


# # ------------对所有Alis2s3_train图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/Ali/train/tamper/'
# val_percent = 0.1
# imgs_name = os.listdir(img_path)
# imgs_name.sort()
# s2s, s3s = [], []
# for img_name in imgs_name:
#     if int(img_name[:-4]) <= 1000:
#         s2s.append(img_name)
#     else:
#         s3s.append(img_name)
# print('s2:', len(s2s))
# print('s3:', len(s3s))
# train_f = open('/pubdata/zhengkengtao/Ali/train/train[{}].txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/Ali/train/val[{}].txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# s2_train = s2s[:round(len(s2s)*train_percent)]
# s3_train = s3s[:round(len(s3s)*train_percent)]
# print('s2_train:', len(s2_train))
# print('s3_train:', len(s3_train))  # 898,之前895
# train = s2_train + s3_train
# # val
# val = imgs_name
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('.jpg', '.png')
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('.jpg', '.png')
#     print(val_name + ',' + val_mask_name, file=val_f)


# # ----------对crop后的图像块选择一定篡改区域比例的图像，再对挑出的图像划分训练验证集，train和val对于每种篡改类型比例一致-----------（运行速度很慢）
# # 1.挑选图像块
# img_path = '/pubdata/zhengkengtao/Ali/train/tamper_crop_128/'
# mask_path = '/pubdata/zhengkengtao/Ali/train/mask_crop_128/'
# imgs_name = os.listdir(img_path)
# masks_name = os.listdir(mask_path)
# use_f = open('/pubdata/zhengkengtao/Ali/train/crop_128_[tamper0-0.5].txt', 'w+')
# all = 128*128
# low_pixel = int(all * 0)
# high_pixel = int(all * 0.5)
# print(low_pixel, high_pixel)
# selected_tampers = []
# selected_masks = []
# for mask_name in masks_name:
#     print(mask_name)
#     mask = Image.open(mask_path + mask_name)
#     mask = mask.convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     # print(mask.min(), mask.max())
#     mask_norm = np.array(mask / 255, dtype=np.uint8)
#     sum_all = sum(sum(i) for i in mask_norm)
#     if sum_all>low_pixel and sum_all<=high_pixel:
#         selected_masks.append(mask_name)
#         tmp = mask_name
#         tamper_name = tmp
#         selected_tampers.append(tamper_name)
#         print(tamper_name + ',' + mask_name, file=use_f)
# # 2.对挑选出的图像块分train和val,train和val对于每种篡改类型比例一致
# file_path = '/pubdata/zhengkengtao/Ali/train/crop_128_[tamper0-0.5].txt'
# selected_tampers = []
# with open(file_path, 'r') as f:
#     for line in f:
#         img = line.strip('\n').split(',')[0]
#         selected_tampers.append(img)
# val_percent = 0.1
# train_percent = 1 - val_percent
# train_f = open('/pubdata/zhengkengtao/Ali/train/crop_128_[tamper0-0.5]_train[{}].txt'.format(train_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/certificate/crop_128_[tamper0-0.5]_val[{}].txt'.format(val_percent), 'w+')
# coms, spls, adds, rems, rmas = [], [], [], [], []
# for tamper in selected_tampers:
#     if '_com' in tamper:
#         coms.append(tamper)
#     elif '_spl' in tamper:
#         spls.append(tamper)
#     elif '_add' in tamper:
#         adds.append(tamper)
#     elif '_rem' in tamper:
#         rems.append(tamper)
#     elif '_rma' in tamper:
#         rmas.append(tamper)
# # train
# com_train = coms[:round(len(coms)*train_percent)]
# spl_train = spls[:round(len(spls)*train_percent)]
# add_train = adds[:round(len(adds)*train_percent)]
# rem_train = rems[:round(len(rems)*train_percent)]
# rma_train = rmas[:round(len(rmas)*train_percent)]
# train = com_train + spl_train + add_train + rem_train + rma_train
# # val
# val = selected_tampers
# for train_name in train:
#     val.remove(train_name)
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('pngps', 'pngms', 1)
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('pngps', 'pngms', 1)
#     print(val_name + ',' + val_mask_name, file=val_f)


# # ------------对所有Ali_new_train图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/tamper/'
# val_percent = 0.1
# imgs_name = os.listdir(img_path)
# imgs_name.sort()
# train_f = open('/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train[{}]_2.txt'.format(1-val_percent), 'w+')
# val_f = open('/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/val[{}]_2.txt'.format(val_percent), 'w+')
# train_percent = 1 - val_percent
# # train
# train = imgs_name[:round(len(imgs_name)*0.1)] + imgs_name[round(len(imgs_name)*0.2):]
# print('train:', len(train))
# # val
# val = imgs_name
# for train_name in train:
#     val.remove(train_name)
# print('val:', len(val))
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('.jpg', '.png')
#     print(train_name + ',' + train_mask_name, file=train_f)
# for val_name in val:
#     val_mask_name = val_name.replace('.jpg', '.png')
#     print(val_name + ',' + val_mask_name, file=val_f)



# # ------------对SUPAT图像划分训练集和验证集，train和val对于每种篡改类型比例一致---------------
# img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/img/'
# gt_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/mask/'
# train_img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split415/train_imgs/'
# train_gt_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split415/train_gt/'
# val_img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split415/val_imgs/'
# val_gt_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split415/val_gt/'
# test_img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split415/test_imgs/'
# test_gt_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split415/test_gt/'
# ns = os.listdir(img_path)
# train = os.listdir(train_img_path)
# # val
# random.shuffle(ns)
# train = ns[:1600]
# val = ns[1600:2000]
# test = ns[2000:]
# # 记录
# for train_name in train:
#     train_mask_name = train_name.replace('jpg', 'png', 1)
#     shutil.copyfile(img_path+train_name, train_img_path+train_name)
#     shutil.copyfile(gt_path+train_mask_name, train_gt_path+train_mask_name)
# for val_name in val:
#     val_mask_name = val_name.replace('jpg', 'png', 1)
#     shutil.copyfile(img_path+val_name, val_img_path+val_name)
#     shutil.copyfile(gt_path+val_mask_name, val_gt_path+val_mask_name)
# for test_name in test:
#     test_mask_name = test_name.replace('jpg', 'png', 1)
#     shutil.copyfile(img_path+test_name, test_img_path+test_name)
#     shutil.copyfile(gt_path+test_mask_name, test_gt_path+test_mask_name)


img_path = '/pubdata/zhengkengtao/docimg/printer/2/img/'
gt_path = '/pubdata/zhengkengtao/docimg/printer/2/gt3/'
train_img_path = '/pubdata/zhengkengtao/docimg/printer/1_2/train_imgs/'
train_gt_path = '/pubdata/zhengkengtao/docimg/printer/1_2/train_gt3/'
val_img_path = '/pubdata/zhengkengtao/docimg/printer/1_2/val_imgs/'
val_gt_path = '/pubdata/zhengkengtao/docimg/printer/1_2/val_gt3/'
ns = os.listdir(img_path)
train = os.listdir(train_img_path)
# val
random.shuffle(ns)
train = ns[:450]
val = ns[450:]
# 记录
for train_name in train:
    train_mask_name = train_name.replace('psc', 'gt3', 1)
    shutil.copyfile(img_path+train_name, train_img_path+train_name)
    shutil.copyfile(gt_path+train_mask_name, train_gt_path+train_mask_name)
for val_name in val:
    val_mask_name = val_name.replace('psc', 'gt3', 1)
    shutil.copyfile(img_path+val_name, val_img_path+val_name)
    shutil.copyfile(gt_path+val_mask_name, val_gt_path+val_mask_name)

import os
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import warnings
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
import shutil
# scikit-image读取和存储格式是RGB，也是numpy.ndarray格式
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


# # ----------裁剪docimg-----------------------------------------------------------------------------------------------
# tamper_path = '/data1/zhengkengtao/docimg/docimg_split811/train_images/'
# mask_path = '/data1/zhengkengtao/docimg/docimg_split811/train_gt3/'
# print(len(os.listdir(mask_path)))
# size, stride = 64, 32 # 裁剪的图像大小, 步长
# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32_1/images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32_1/gt3/'
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# # imgs_name = imgs_name[0*100: 1*100]
# file_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/croptrainlog.txt'
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = io.imread(tamper_path + img_name)
#         mask_name = img_name.replace('psc', 'gt3', 1)
#         mask = io.imread(mask_path + mask_name)
#         h, w = img.shape[0], img.shape[1]
#         h_num = int((h - size) / stride + 1)
#         w_num = int((w - size) / stride + 1)
#         h_lable = False
#         w_lable = False
#         h_max = (h_num - 1) * stride + size
#         w_max = (w_num - 1) * stride + size
#         assert h_max <= h
#         assert w_max <= w
#         if h_max < h:
#             h_lable = True
#         if w_max < w:
#             w_lable = True
#         num = 0
#         for i in range(h_num + int(h_lable)):
#             for j in range(w_num + int(w_lable)):
#                 num = num + 1
#                 h1 = i * stride
#                 h2 = i * stride + size
#                 w1 = j * stride
#                 w2 = j * stride + size
#                 if i == h_num:
#                     h1 = h - size
#                     h2 = h
#                 if j == w_num:
#                     w1 = w - size
#                     w2 = w
#                 patch = img[h1:h2, w1:w2, :]
#                 mask_patch = mask[h1:h2, w1:w2, :]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


# # Orig为一类，Mosaic和Tamper为另一类，挑选出real和tamper
# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/gt3/'
# select_tamper_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/select/tamper/'
# select_real_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/select/real/'
# if (os.path.exists(select_tamper_dir) == False): os.makedirs(select_tamper_dir)
# if (os.path.exists(select_real_dir) == False): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# # imgs_name = imgs_name[0*600000:1*600000]
# for img_name in tqdm(imgs_name):
#     print(img_name + 'copyfile')
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('psc', 'gt3', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if 76 in mask or 29 in mask:
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     elif 76 not in mask and 29 not in mask:
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)


# # Orig和Mosaic视为一类，Tamper为另一类，挑选出real和tamper
# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/test_images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_all_size64_stride32/test_gt3/'
# select_tamper_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/test_tamper/'
# select_real_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/test_real/'
# if (os.path.exists(select_tamper_dir) == False): os.makedirs(select_tamper_dir)
# if (os.path.exists(select_real_dir) == False): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# imgs_name = imgs_name[0*400000:1*400000] # 1347129
# for img_name in tqdm(imgs_name):
#     print(img_name + 'copyfile')
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('psc', 'gt3', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if 76 in mask:
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)


# # ----------裁剪Alinew-----------------------------------------------------------------------------------------------
# # tamper_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_imgs/'
# # mask_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
# tamper_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_imgs/'
# mask_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_gt/'
# print(len(os.listdir(mask_path)))
# size, stride = 64, 32 # 裁剪的图像大小, 步长
# # tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/test_imgs/'
# # mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/test_gt/'
# # tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/img/'
# # mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/mask/'
# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/test_imgs_left/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/test_gt_left/'
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# tamper_crop_names = os.listdir(tamper_crop_path)
# print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# # imgs_name = imgs_name[0*100: 1*100]
# # file_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/crop64_test_log.txt'
# file_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/croplog.txt'
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# print(len(imgs_name))
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = io.imread(tamper_path + img_name)
#         mask_name = img_name.replace('jpg', 'png', 1)
#         mask = io.imread(mask_path + mask_name)
#         h, w = img.shape[0], img.shape[1]
#         h_num = int((h - size) / stride + 1)
#         w_num = int((w - size) / stride + 1)
#         h_lable = False
#         w_lable = False
#         h_max = (h_num - 1) * stride + size
#         w_max = (w_num - 1) * stride + size
#         assert h_max <= h
#         assert w_max <= w
#         if h_max < h:
#             h_lable = True
#         if w_max < w:
#             w_lable = True
#         num = 0
#         for i in range(h_num + int(h_lable)):
#             for j in range(w_num + int(w_lable)):
#                 num = num + 1
#                 h1 = i * stride
#                 h2 = i * stride + size
#                 w1 = j * stride
#                 w2 = j * stride + size
#                 if i == h_num:
#                     h1 = h - size
#                     h2 = h
#                 if j == w_num:
#                     w1 = w - size
#                     w2 = w
#                 patch = img[h1:h2, w1:w2, :]
#                 if len(mask.shape) == 2:
#                     mask_patch = mask[h1:h2, w1:w2]
#                 else:
#                     mask_patch = mask[h1:h2, w1:w2, :]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/test_imgs_left/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/test_gt_left/'
# select_tamper_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/select/test_tamper/'
# select_real_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/select/test_real/'
# if not os.path.exists(select_tamper_dir): os.makedirs(select_tamper_dir)
# if not os.path.exists(select_real_dir): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# L = len(imgs_name)
# print(len(imgs_name))
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('jpg', 'png', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if np.any(mask):  # 有非0元素输出True，全为0输出False
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)
#
# print(len(os.listdir(tamper_crop_path)))
# print(len(os.listdir(select_tamper_dir)))
# print(len(os.listdir(select_real_dir)))
#
#
# # tamper_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# # img_name = 'psc_honor30_oc_110_rma.png'
# # img = io.imread(tamper_path + img_name)
# # mask_name = img_name.replace('jpg', 'png', 1)
# # size = 64
# # stride = 32
# # h, w = img.shape[0], img.shape[1]
# # h_num = int((h - size) / stride + 1)
# # w_num = int((w - size) / stride + 1)
# # h_lable = False
# # w_lable = False
# # h_max = (h_num - 1) * stride + size
# # w_max = (w_num - 1) * stride + size
# # assert h_max <= h
# # assert w_max <= w
# # if h_max < h:
# #     h_lable = True
# # if w_max < w:
# #     w_lable = True
# # num = 0
# # print(h_num + int(h_lable))
# # print(w_num + int(w_lable))
#

# # ----------裁剪supat----------------------------------------------------------------------------------------------
# tamper_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/test_imgs/'
# mask_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/test_gt/'
# print(len(os.listdir(mask_path)))
# size, stride = 64, 32 # 裁剪的图像大小, 步长
# # tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/test_imgs/'
# # mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/test_gt/'
# # tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/img/'
# # mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/mask/'
# tamper_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/test_imgs_crop_64x64/'
# mask_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/test_gt_crop_64x64/'
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# tamper_crop_names = os.listdir(tamper_crop_path)
# print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# # imgs_name = imgs_name[0*100: 1*100]
# # file_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/crop64_test_log.txt'
# file_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/crop64_test_log.txt'
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# print(len(imgs_name))
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = io.imread(tamper_path + img_name)
#         mask_name = img_name.replace('tif', 'png', 1)
#         mask = io.imread(mask_path + mask_name)
#         h, w = img.shape[0], img.shape[1]
#         h_num = int((h - size) / stride + 1)
#         w_num = int((w - size) / stride + 1)
#         h_lable = False
#         w_lable = False
#         h_max = (h_num - 1) * stride + size
#         w_max = (w_num - 1) * stride + size
#         assert h_max <= h
#         assert w_max <= w
#         if h_max < h:
#             h_lable = True
#         if w_max < w:
#             w_lable = True
#         num = 0
#         for i in range(h_num + int(h_lable)):
#             for j in range(w_num + int(w_lable)):
#                 num = num + 1
#                 h1 = i * stride
#                 h2 = i * stride + size
#                 w1 = j * stride
#                 w2 = j * stride + size
#                 if i == h_num:
#                     h1 = h - size
#                     h2 = h
#                 if j == w_num:
#                     w1 = w - size
#                     w2 = w
#                 patch = img[h1:h2, w1:w2, :]
#                 if len(mask.shape) == 2:
#                     mask_patch = mask[h1:h2, w1:w2]
#                 else:
#                     mask_patch = mask[h1:h2, w1:w2, :]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


# tamper_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/test_imgs_crop_64x64/'
# mask_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/test_gt_crop_64x64/'
# select_tamper_dir = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/test_tamper/'
# select_real_dir = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/test_real/'
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# # imgs_name = imgs_name[0*10000:1*10000]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('psc', 'gt3', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if np.sum(mask) != 0:
#         print(1)
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         print(0)
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)


# ----------裁剪Alinew----------------------------------------------------------------------------------------------
tamper_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/test_imgs/'
mask_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/test_gt/'
print(len(os.listdir(mask_path)))
size, stride = 64, 32 # 裁剪的图像大小, 步长
# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/test_imgs/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/crop_64x64/test_gt/'
# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/img/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/crop_64x64/mask/'
tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/test_imgs_crop_64x64/'
mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/test_gt_crop_64x64/'
if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
tamper_crop_names = os.listdir(tamper_crop_path)
print(len(tamper_crop_names))
imgs_name = os.listdir(tamper_path)
imgs_name.sort()
imgs_name = imgs_name[0*800: 1*800]
file_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/crop64stride32_test_log.txt'
with open(file_path, 'r') as f:
    for line in f:
        cropped_img = line.strip('\n')
        imgs_name.remove(cropped_img)
# 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
print(len(imgs_name))
with open(file_path, 'a') as f:
    for img_name in tqdm(imgs_name):
        print(img_name)
        img = io.imread(tamper_path + img_name)
        mask_name = img_name.replace('jpg', 'png', 1)
        mask = io.imread(mask_path + mask_name)
        h, w = img.shape[0], img.shape[1]
        h_num = int((h - size) / stride + 1)
        w_num = int((w - size) / stride + 1)
        h_lable = False
        w_lable = False
        h_max = (h_num - 1) * stride + size
        w_max = (w_num - 1) * stride + size
        assert h_max <= h
        assert w_max <= w
        if h_max < h:
            h_lable = True
        if w_max < w:
            w_lable = True
        num = 0
        for i in range(h_num + int(h_lable)):
            for j in range(w_num + int(w_lable)):
                num = num + 1
                h1 = i * stride
                h2 = i * stride + size
                w1 = j * stride
                w2 = j * stride + size
                if i == h_num:
                    h1 = h - size
                    h2 = h
                if j == w_num:
                    w1 = w - size
                    w2 = w
                patch = img[h1:h2, w1:w2, :]
                if len(mask.shape) == 2:
                    mask_patch = mask[h1:h2, w1:w2]
                else:
                    mask_patch = mask[h1:h2, w1:w2, :]
                io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
                io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
        f.write(img_name + '\n')


# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/DIDNetcrop/test_imgs_crop_64x64/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/DIDNetcrop/test_gt_crop_64x64/'
# select_tamper_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/DIDNetcrop/test_tamper/'
# select_real_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/DIDNetcrop/test_real/'
# if not os.path.exists(select_tamper_dir): os.makedirs(select_tamper_dir)
# if not os.path.exists(select_real_dir): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# # imgs_name = imgs_name[0*10000:1*10000]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('psc', 'gt3', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if np.sum(mask) != 0:
#         print(1)
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         print(0)
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)
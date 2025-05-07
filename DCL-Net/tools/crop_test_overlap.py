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


# ----------裁剪docimg-----------------------------------------------------------------------------------------------
tamper_path = '/pubdata/zhengkengtao/docimg/printer/3_4/img/'
mask_path = '/pubdata/zhengkengtao/docimg/printer/3_4/gt3/'
print(len(os.listdir(mask_path)))
size, stride = 512, 256 # 裁剪的图像大小, 步长
tamper_crop_path = '/pubdata/zhengkengtao/docimg/printer/3_4/img_crop{}stride{}/'.format(size, stride)
mask_crop_path = '/pubdata/zhengkengtao/docimg/printer/3_4/gt3_crop{}stride{}/'.format(size, stride)
if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# tamper_crop_names = os.listdir(tamper_crop_path)
# print(len(tamper_crop_names))
imgs_name = os.listdir(tamper_path)
imgs_name.sort()
imgs_name = imgs_name[0*50: 1*50]
file_path = '/pubdata/zhengkengtao/docimg/printer/3_4/crop{}stride{}_log.txt'.format(size, stride)
with open(file_path, 'r') as f:
    for line in f:
        cropped_img = line.strip('\n')
        imgs_name.remove(cropped_img)
# 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
with open(file_path, 'a') as f:
    for img_name in tqdm(imgs_name):
        print(img_name)
        img = Image.open(tamper_path + img_name).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        mask_name = img_name.replace('psc', 'gt3', 1)
        mask = Image.open(mask_path + mask_name).convert('RGB')
        mask = np.array(mask, dtype=np.uint8)
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
                mask_patch = mask[h1:h2, w1:w2, :]
                io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
                io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
        f.write(img_name + '\n')


# # ----------裁剪docimg-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_tamper/'
# # mask_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaicGT3/'
# # print(len(os.listdir(mask_path)))
# size, stride = 512, int(512/2) # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_tamper_crop{}stride{}/'.format(size, stride)
# # mask_crop_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaicGT3_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# # if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*40: 1*40]
# file_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_tamper_crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         # mask_name = img_name.replace('psc', 'gt3', 1)
#         # mask_name = mask_name.replace('jpg', 'png', 1)
#         # mask = Image.open(mask_path + mask_name).convert('RGB')
#         # mask = np.array(mask, dtype=np.uint8)
#         h, w = img.shape[0], img.shape[1]
#         h0, w0 = img.shape[0], img.shape[1]
#         if h < 512 and w >= 512:
#             if w == 512:
#                 w = 512 + 64
#             img = cv2.resize(img, (w, 512))
#         elif w < 512 and h >= 512:
#             if h == 512:
#                 h = 512 + 64
#             img = cv2.resize(img, (512, h))
#         elif h < 512 and w < 512:
#             img = cv2.resize(img, (512, 512 + 64))
#         elif h == 512 and w == 512:
#             img = cv2.resize(img, (512, 512 + 64))  # 避免只才裁剪出来一块，训练时BN报错
#         else:
#             img = img
#         h, w = img.shape[0], img.shape[1]
#         # if h != h0 or w != w0:
#         #     # maskL = Image.open(mask_path + mask_name).convert('L')
#         #     newgt3 = maskL
#         #     newgt3[maskL == 255] = 0
#         #     newgt3[maskL == 76] = 1
#         #     newgt3[maskL == 29] = 2
#         #     dstgt3 = cv2.resize(newgt3, (w, h))
#         #     dstgt3 = np.uint8(dstgt3)
#         #     pred3 = np.ones([h, w, 3]) * 255
#         #     pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
#         #     pred3_1[dstgt3 == 1] = 0
#         #     pred3_2[dstgt3 == 1] = 0
#         #     pred3_0[dstgt3 == 2] = 0
#         #     pred3_1[dstgt3 == 2] = 0
#         #     pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
#         #     pred3 = Image.fromarray(np.uint8(pred3))
#         #     pred3 = pred3.convert('RGB')
#         #     mask = np.array(pred3, np.uint8)
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
#                 # mask_patch = mask[h1:h2, w1:w2, :]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 # io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


# # ----------裁剪docimg_testorig-----------------------------------------------------------------------------------------------
# img_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig/'
# print(len(os.listdir(img_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# img_crop_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(img_crop_path): os.makedirs(img_crop_path)
# imgs_name = os.listdir(img_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*40: 1*40]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = Image.open(img_path + img_name).convert('RGB')
#     img = np.array(img, dtype=np.uint8)
#     h, w = img.shape[0], img.shape[1]
#     h_num = int((h - size) / stride + 1)
#     w_num = int((w - size) / stride + 1)
#     h_lable = False
#     w_lable = False
#     h_max = (h_num - 1) * stride + size
#     w_max = (w_num - 1) * stride + size
#     assert h_max <= h
#     assert w_max <= w
#     if h_max < h:
#         h_lable = True
#     if w_max < w:
#         w_lable = True
#     num = 0
#     for i in range(h_num + int(h_lable)):
#         for j in range(w_num + int(w_lable)):
#             num = num + 1
#             h1 = i * stride
#             h2 = i * stride + size
#             w1 = j * stride
#             w2 = j * stride + size
#             if i == h_num:
#                 h1 = h - size
#                 h2 = h
#             if j == w_num:
#                 w1 = w - size
#                 w2 = w
#             patch = img[h1:h2, w1:w2, :]
#             io.imsave(img_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)


# # ----------裁剪supatlantique-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper/'
# mask_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/mask/'
# print(len(os.listdir(mask_path)))
# size, stride = 256, 128 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/mask_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*40: 1*40]
# file_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('tif', 'png', 1)
#         mask = Image.open(mask_path + mask_name).convert('L')
#         mask = np.array(mask, dtype=np.uint8)
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
#                 mask_patch = mask[h1:h2, w1:w2]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


# # ----------裁剪supatlantique-OrigMosaic-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/orig_mosaic/MeituMosaic/'
# mask_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/orig_mosaic/MeituMosaicGT3/'
# print(len(os.listdir(mask_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/orig_mosaic/MeituMosaic_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/orig_mosaic/MeituMosaicGT3_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*55: 1*55]
# file_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/orig_mosaic/MeituMosaic_crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('tif', 'png', 1)
#         mask_name = mask_name.replace('mosaic', 'gt3', 1)
#         mask = Image.open(mask_path + mask_name).convert('RGB')
#         mask = np.array(mask, dtype=np.uint8)
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


# # ----------裁剪docimg_czw-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/docforgery_czw/mobilephone/tamper_png/'
# mask_path = '/pubdata/zhengkengtao/docforgery_czw/mobilephone/mask/'
# print(len(os.listdir(mask_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/docforgery_czw/mobilephone/tamper_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/docforgery_czw/mobilephone/mask_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*30: 1*30]
# file_path = '/pubdata/zhengkengtao/docforgery_czw/mobilephone/crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('forgery', 'gt', 1)
#         mask = Image.open(mask_path + mask_name).convert('L')
#         mask = np.array(mask, dtype=np.uint8)
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
#                 mask_patch = mask[h1:h2, w1:w2]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


# # ----------裁剪findit T2tamper-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/findit/T2tamper/'
# mask_path = '/pubdata/zhengkengtao/findit/T2gt/'
# print(len(os.listdir(mask_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/findit/T2tamper_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/findit/T2gt_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*60: 1*60]
# file_path = '/pubdata/zhengkengtao/findit/crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('jpg', 'png', 1)
#         mask = Image.open(mask_path + mask_name).convert('L')
#         mask = np.array(mask, dtype=np.uint8)
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
#                 mask_patch = mask[h1:h2, w1:w2]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')


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


# # ----------裁剪PS_arbitrary-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/PS_arbitrary/tamper/'
# mask_path = '/pubdata/zhengkengtao/PS_arbitrary/mask/'
# print(len(os.listdir(mask_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/PS_arbitrary/tamper_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/PS_arbitrary/mask_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[330: 420]
# file_path = '/pubdata/zhengkengtao/PS_arbitrary/crop{}stride{}_log.txt'.format(size, stride)
# # with open(file_path, 'r') as f:
# #     for line in f:
# #         cropped_img = line.strip('\n')
# #         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('jpg', 'png', 1)
#         mask_name = mask_name.replace('ps', 'ms', 1)
#         mask = Image.open(mask_path + mask_name).convert('L')
#         mask = np.array(mask, dtype=np.uint8)
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
#                 mask_patch = mask[h1:h2, w1:w2]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')



# # ----------裁剪findit_4types-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/findit/findit_4types/imgs/'
# mask_path = '/pubdata/zhengkengtao/findit/findit_4types/gt3/'
# print(len(os.listdir(mask_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/findit/findit_4types/imgs_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/findit/findit_4types/gt3_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*80: 1*80]
# file_path = '/pubdata/zhengkengtao/findit/findit_4types/imgs_crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         if cropped_img in imgs_name:
#             imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('jpg', 'png')
#         mask_name = 'gt3_' + mask_name
#         mask = Image.open(mask_path + mask_name).convert('RGB')
#         mask = np.array(mask, dtype=np.uint8)
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


# # ----------裁剪PS_arbitrary-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/PS_arbitrary/tamper/'
# mask_path = '/pubdata/zhengkengtao/PS_arbitrary/mask/'
# print(len(os.listdir(mask_path)))
# size, stride = 512, 256 # 裁剪的图像大小, 步长
# tamper_crop_path = '/pubdata/zhengkengtao/PS_arbitrary/tamper_crop{}stride{}/'.format(size, stride)
# mask_crop_path = '/pubdata/zhengkengtao/PS_arbitrary/mask_crop{}stride{}/'.format(size, stride)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[1*500: ]
# file_path = '/pubdata/zhengkengtao/PS_arbitrary/crop{}stride{}_log.txt'.format(size, stride)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = Image.open(tamper_path + img_name).convert('RGB')
#         img = np.array(img, dtype=np.uint8)
#         mask_name = img_name.replace('jpg', 'png', 1)
#         mask_name = mask_name.replace('ps', 'ms', 1)
#         mask = Image.open(mask_path + mask_name).convert('L')
#         mask = np.array(mask, dtype=np.uint8)
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
#                 mask_patch = mask[h1:h2, w1:w2]
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
#         f.write(img_name + '\n')









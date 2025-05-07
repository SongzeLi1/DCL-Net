#coding=utf-8
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def gaussian_blur(img,mask = None):
    blur_kernel = random.choice([3, 5, 7, 9, 11])
    img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), sigmaX=0, sigmaY=0)
    return img,mask
def median_blur(img, blur_kernel = 3):
    img = cv2.medianBlur(img, blur_kernel)
    return img
def mean_blur(img,mask = None):
    blur_kernel = random.choice([3, 5, 7, 9, 11])
    img = cv2.blur(img, (blur_kernel, blur_kernel))
    return img,mask
def add_gaussian_noise(img, loc=0, scale=2):
    H, W, C = img.shape
    # N = np.random.randint(10, 100) / 10. * np.random.normal(loc=0, scale=1, size=(H, W, 1))
    N = 1. * np.random.normal(loc=loc, scale=scale, size=(H, W, 1)) # 均值和标准差
    N = np.repeat(N, C, axis=2)
    img = img.astype(np.int32)
    img = N + img
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img
def JPEG_compression(img,mask = None):
    compression_quality = random.randint(70, 100)
    result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
    img = cv2.imdecode(encimg, 1)
    return img, mask


# # 压缩
# print(len(os.listdir('/data1/zhengkengtao/docimg/docimg_split811/test_images_jpg/jpg60/')))
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_jpg/jpg65/'
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     quality = 65 # 左右都包括
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     cv2.imwrite(dst_path + img_name[:-4] + '_qf65.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir('/data1/zhengkengtao/docimg/docimg_split811/test_images_jpg/jpg65/')))


# 压缩
path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic/'
dst_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic_jpg80/'
if not os.path.exists(dst_path): os.makedirs(dst_path)
img_names = os.listdir(path)
for img_name in img_names:
    print(img_name)
    quality = 80
    img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
    img_name = img_name.replace('mosaic_', 'psc_')
    cv2.imwrite(dst_path + img_name[:-4] + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])


# # # 高斯加噪
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# mean = 0
# stds = [5, 10, 15, 20, 25, 30]
# for std in stds:
#     print('***********', std, '******************')
#     dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/gaussnoise/mean{}std{}/'.format(mean, std)
#     if not os.path.exists(dst_path): os.makedirs(dst_path)
#     img_names = os.listdir(path)
#     for img_name in img_names:
#         print(img_name)
#         img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#         img = np.array(img, dtype=np.uint8)
#         dstimg = add_gaussian_noise(img, loc=mean, scale=std)
#         cv2.imwrite(dst_path + img_name[:-4] + '.png', dstimg)
#     print(len(os.listdir(dst_path)))


# # # 中值滤波模糊
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# blur_kernels = [25]
# for blur_kernel in blur_kernels:
#     dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/medianblur/blurkernel{}/'.format(blur_kernel)
#     if not os.path.exists(dst_path): os.makedirs(dst_path)
#     img_names = os.listdir(path)
#     for img_name in img_names:
#         print(img_name)
#         img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#         img = np.array(img, dtype=np.uint8)
#         dstimg = median_blur(img, blur_kernel)
#         cv2.imwrite(dst_path + img_name[:-4] + '.png', dstimg)
#     print(len(os.listdir(dst_path)))


# # # resize
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# rates = [0.3, 0.7]
# for rate in rates:
#     dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/resize/rate{}/'.format(rate)
#     if not os.path.exists(dst_path): os.makedirs(dst_path)
#     img_names = os.listdir(path)
#     for img_name in img_names:
#         print(img_name)
#         img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#         img = np.array(img, dtype=np.uint8)
#         h, w = img.shape[0], img.shape[1]
#         h, w = round(h * rate), round(w * rate)
#         dstimg = cv2.resize(img, (w, h))
#         cv2.imwrite(dst_path + img_name[:-4] + '.png', dstimg)
#     print(len(os.listdir(dst_path)))
#
#
# # # resize_gt
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# rates = [0.3, 0.7]
# for rate in rates:
#     dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/resize_gt3/rate{}/'.format(rate)
#     if not os.path.exists(dst_path): os.makedirs(dst_path)
#     img_names = os.listdir(path)
#     for img_name in img_names:
#         print(img_name)
#         img = cv2.imread(path + img_name, cv2.IMREAD_GRAYSCALE)
#         img = np.array(img, dtype=np.uint8)
#         # print(img.min(),img.max())
#         h, w = img.shape
#         newimg = img
#         newimg[img==255]=0
#         newimg[img==76]=1
#         newimg[img==29]=2
#         new_h, new_w = round(h * rate), round(w * rate)
#         dstimg = cv2.resize(newimg, (new_w, new_h))
#         dstimg = np.uint8(dstimg)
#         # for i in range(h):
#         #     for j in range(w):
#         #         if (dstimg[i,j]!=0) and (dstimg[i,j]!=1) and (dstimg[i,j]!=2):
#         #             print(dstimg[i,j])
#         pred3 = np.ones([new_h, new_w, 3]) * 255
#         pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
#         pred3_1[dstimg == 1] = 0
#         pred3_2[dstimg == 1] = 0
#         pred3_0[dstimg == 2] = 0
#         pred3_1[dstimg == 2] = 0
#         pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
#         pred3 = Image.fromarray(np.uint8(pred3))
#         pred3 = pred3.convert('RGB')
#         pred3.save(dst_path + img_name[:-4] + '.png')
#     print(len(os.listdir(dst_path)))


    #     h, w = img.shape[0], img.shape[1]
    #     h, w = round(h * rate), round(w * rate)
    #     dstimg = cv2.resize(img, (w, h))
    #     dstimgnp = np.array(dstimg, dtype=np.uint8)
    #     cv2.imwrite(dst_path + img_name[:-4] + '.png', dstimg)
    # print(len(os.listdir(dst_path)))

# path  = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/resize_gt3/rate0.5/'
# for i in os.listdir(path):
#     img = Image.open(path + i).convert('L')
#     img = np.array(img, dtype=np.uint8)
#     h,w = img.shape
#     for i in range(h):
#         for j in range(w):
#             if (img[i,j]!=255) and (img[i,j]!=76) and (img[i,j]!=29):
#                 print(img[i,j])
#     # print(img.min(), img.max())


# # 中值滤波模糊+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# blur_kernel = 5
# quality = 80
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/medianblur_jpg/blurkernel{}_jpg{}/'.format(blur_kernel, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     dstimg = median_blur(img, blur_kernel)
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))


# # 高斯噪声+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# mean = 0
# std = 10
# quality = 80
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/gaussnoise_jpg/mean{}std{}_jpg{}/'.format(mean, std, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     dstimg = add_gaussian_noise(img, loc=mean, scale=std)
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))


# # resize+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# rate = 0.5
# quality = 80
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/resize_jpg/rate{}_jpg{}/'.format(rate, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     h, w = img.shape[0], img.shape[1]
#     h, w = round(h * rate), round(w * rate)
#     dstimg = cv2.resize(img, (w, h))
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))


# # 中值滤波模糊+高斯噪声+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# blur_kernel = 5
# mean = 0
# std = 10
# quality = 80
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/medianblur_gaussnoise_jpg/blurkernel{}_mean{}std{}_jpg{}/'.format(blur_kernel, mean, std, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     dstimg = median_blur(img, blur_kernel)
#     dstimg = add_gaussian_noise(dstimg, loc=mean, scale=std)
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))


# # 中值滤波模糊+resize+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# blur_kernel = 5
# rate = 0.5
# quality = 80
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/medianblur_resize_jpg/blurkernel{}_rate{}_jpg{}/'.format(blur_kernel, rate, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     dstimg = median_blur(img, blur_kernel)
#     h, w = img.shape[0], img.shape[1]
#     h, w = round(h * rate), round(w * rate)
#     dstimg = cv2.resize(dstimg, (w, h))
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))
#
#
# # 高斯噪声+resize+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# mean = 0
# std = 10
# rate = 0.5
# quality = 80
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/gaussnoise_resize_jpg/mean{}std{}_rate{}_jpg{}/'.format(mean, std, rate, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     dstimg = add_gaussian_noise(img, loc=mean, scale=std)
#     h, w = img.shape[0], img.shape[1]
#     h, w = round(h * rate), round(w * rate)
#     dstimg = cv2.resize(dstimg, (w, h))
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))


# # 中值滤波模糊+高斯噪声+resize+JPEG压缩
# path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
# blur_kernel = 5
# mean = 0
# std = 10
# rate = 0.5
# quality = 90
# dst_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images_robust/medianblur_gaussnoise_resize_jpg/blurkernel{}_mean{}std{}_rate{}_jpg{}/'.format(blur_kernel, mean, std, rate, quality)
# if not os.path.exists(dst_path): os.makedirs(dst_path)
# img_names = os.listdir(path)
# for img_name in img_names:
#     print(img_name)
#     img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
#     img = np.array(img, dtype=np.uint8)
#     dstimg = median_blur(img, blur_kernel)
#     dstimg = add_gaussian_noise(dstimg, loc=mean, scale=std)
#     h, w = img.shape[0], img.shape[1]
#     h, w = round(h * rate), round(w * rate)
#     dstimg = cv2.resize(dstimg, (w, h))
#     cv2.imwrite(dst_path + img_name[:-4] + '.jpg', dstimg, [cv2.IMWRITE_JPEG_QUALITY, quality])
# print(len(os.listdir(dst_path)))
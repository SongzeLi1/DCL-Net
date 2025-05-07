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


# # ----------裁剪-----------------------------------------------------------------------------------------------
# tamper_path = '/data1/zhengkengtao/docimg/docimg_split811/val_images/'
# mask_path = '/data1/zhengkengtao/docimg/docimg_split811/val_gt3/'
# print(len(os.listdir(mask_path)))
# d = 64 # 裁剪的图像大小
# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/val_images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/val_gt3/'
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*100:1*100]
# file_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/crop64_val_log1.txt'
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = io.imread(tamper_path + img_name)
#         # print('img.shape:', img.shape)
#         mask_name = img_name.replace('psc', 'gt3', 1)
#         mask = io.imread(mask_path + mask_name)
#         # print('mask.shape', mask.shape)
#         # print('mask.shape:', mask.shape)
#         h, w = img.shape[0], img.shape[1]
#         h_num = h // d
#         w_num = w // d
#         # print(h_num)
#         # print(w_num)
#         # print(h%d, w%d)
#         for i in range(h_num):
#             for j in range(w_num):
#                 if len(img.shape) == 2:
#                     tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d]
#                 else:
#                     tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d]
#                 else:
#                     mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 # print(mask_block.shape)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_block)
#         # print('{} finish crop!'.format(img_name))
#         f.write(img_name + '\n')
#
#
# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/train_images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/train_gt3/'
# select_tamper_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/train_tamper/'
# select_real_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/train_real/'
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# imgs_name = imgs_name[0*1300000:1*1300000]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('psc', 'gt3', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if 76 in mask or 29 in mask:
#         print(1)
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     elif 76 not in mask and 29 not in mask:
#         print(0)
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)



# # ----------裁剪Alinew-----------------------------------------------------------------------------------------------
# tamper_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/train_imgs/'
# mask_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/train_gt/'
# print(len(os.listdir(mask_path)))
# d = 64 # 裁剪的图像大小
# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/train_imgs/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/train_gt/'
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
#
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# file_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/crop64_train_log.txt'
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = io.imread(tamper_path + img_name)
#         # print('img.shape:', img.shape)
#         mask_name = img_name.replace('jpg', 'png', 1)
#         mask = io.imread(mask_path + mask_name)
#         # print('mask.shape', mask.shape)
#         # print('mask.shape:', mask.shape)
#         h, w = img.shape[0], img.shape[1]
#         h_num = h // d
#         w_num = w // d
#         # print(h_num)
#         # print(w_num)
#         # print(h%d, w%d)
#         for i in range(h_num):
#             for j in range(w_num):
#                 if len(img.shape) == 2:
#                     tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d]
#                 else:
#                     tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d]
#                 else:
#                     mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 # print(mask_block.shape)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_block)
#         # print('{} finish crop!'.format(img_name))
#         f.write(img_name + '\n')


# # Orig为一类，Mosaic和Tamper为另一类，挑选出real和tamper
# tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/train_imgs/'
# mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/train_gt/'
# select_tamper_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/select/train_tamper/'
# select_real_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/select/train_real/'
# if not os.path.exists(select_tamper_dir): os.makedirs(select_tamper_dir)
# if not os.path.exists(select_real_dir): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# for img_name in tqdm(imgs_name):
#     if img_name not in os.listdir(select_tamper_dir) and img_name not in os.listdir(select_real_dir):
#         print(img_name)
#         img = io.imread(tamper_crop_path + img_name)
#         mask_name = img_name.replace('jpg', 'png', 1)
#         mask = Image.open(mask_crop_path + mask_name).convert('L')
#         mask = np.array(mask, dtype=np.uint8)
#         if np.any(mask):  # 有非0元素输出True，全为0输出False
#             shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#         else:
#             shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)
#
#
# print(len(os.listdir(tamper_crop_path)))
# print(len(os.listdir(select_tamper_dir)))
# print(len(os.listdir(select_real_dir)))


# # Orig和Mosaic视为一类，Tamper为另一类，挑选出real和tamper
# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/train_images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/train_gt3/'
# select_tamper_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/train_tamper/'
# select_real_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/train_real/'
# if (os.path.exists(select_tamper_dir) == False): os.makedirs(select_tamper_dir)
# if (os.path.exists(select_real_dir) == False): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))  # 2662669
# imgs_name = imgs_name[0*600000:1*600000]
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


# tamper_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/train_images/'
# mask_crop_path = '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/train_gt3/'
# select_tamper_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/select/train_tamper/'
# select_real_dir = '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/select/train_real/'
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# imgs_name = imgs_name[0*10000:1*10000]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name.replace('psc', 'gt3', 1)
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if 76 in mask:
#         print(1)
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         print(0)
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)


# # ----------裁剪supat-----------------------------------------------------------------------------------------------
# tamper_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/train_imgs/'
# mask_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/train_gt/'
# print(len(os.listdir(mask_path)))
# d = 64 # 裁剪的图像大小
# tamper_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/train_imgs_crop_64x64/'
# mask_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/train_gt_crop_64x64/'
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
#
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[:15]
# file_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/crop64_train_log.txt'
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         print(img_name)
#         img = io.imread(tamper_path + img_name)
#         # print('img.shape:', img.shape)
#         mask_name = img_name.replace('tif', 'png', 1)
#         mask = io.imread(mask_path + mask_name)
#         # print('mask.shape', mask.shape)
#         # print('mask.shape:', mask.shape)
#         h, w = img.shape[0], img.shape[1]
#         h_num = h // d
#         w_num = w // d
#         # print(h_num)
#         # print(w_num)
#         # print(h%d, w%d)
#         for i in range(h_num):
#             for j in range(w_num):
#                 if len(img.shape) == 2:
#                     tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d]
#                 else:
#                     tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d]
#                 else:
#                     mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 # print(mask_block.shape)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_block)
#         # print('{} finish crop!'.format(img_name))
#         f.write(img_name + '\n')
#
#
# tamper_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/train_imgs_crop_64x64/'
# mask_crop_path = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/train_gt_crop_64x64/'
# select_tamper_dir = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/train_tamper/'
# select_real_dir = '/data1/zhengkengtao/SUPATLANTIQUE/split37/DIDNetcrop/train_real/'
# if not os.path.exists(select_tamper_dir): os.makedirs(select_tamper_dir)
# if not os.path.exists(select_real_dir): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# # imgs_name = imgs_name[0*10000:1*10000]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = img_name
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     if np.sum(mask) != 0:
#         print(1)
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         print(0)
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)


# ----------裁剪Alinew-----------------------------------------------------------------------------------------------
tamper_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/train_imgs/'
mask_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/train_gt/'
print(len(os.listdir(mask_path)))
d = 64 # 裁剪的图像大小
tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/train_imgs_crop_64x64/'
mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/train_gt_crop_64x64/'
if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)

# tamper_crop_names = os.listdir(tamper_crop_path)
# print(len(tamper_crop_names))
imgs_name = os.listdir(tamper_path)
imgs_name.sort()
# imgs_name = imgs_name[0*800:1*800]
file_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/crop64_train_log.txt'
with open(file_path, 'r') as f:
    for line in f:
        cropped_img = line.strip('\n')
        imgs_name.remove(cropped_img)
# 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
with open(file_path, 'a') as f:
    for img_name in tqdm(imgs_name):
        print(img_name)
        img = io.imread(tamper_path + img_name)
        # print('img.shape:', img.shape)
        mask_name = img_name.replace('jpg', 'png', 1)
        mask = io.imread(mask_path + mask_name)
        # print('mask.shape', mask.shape)
        # print('mask.shape:', mask.shape)
        h, w = img.shape[0], img.shape[1]
        h_num = h // d
        w_num = w // d
        # print(h_num)
        # print(w_num)
        # print(h%d, w%d)
        for i in range(h_num):
            for j in range(w_num):
                if len(img.shape) == 2:
                    tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d]
                else:
                    tamper_block = img[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
                tamper_block = np.array(tamper_block, dtype=np.uint8)
                io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), tamper_block)
                if len(mask.shape) == 2:
                    mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d]
                else:
                    mask_block = mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :]
                mask_block = np.array(mask_block, dtype=np.uint8)
                # print(mask_block.shape)
                io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_block)
        # print('{} finish crop!'.format(img_name))
        f.write(img_name + '\n')


tamper_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/train_imgs_crop_64x64/'
mask_crop_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/train_gt_crop_64x64/'
select_tamper_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/train_tamper/'
select_real_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/DIDNetcrop/train_real/'
if not os.path.exists(select_tamper_dir): os.makedirs(select_tamper_dir)
if not os.path.exists(select_real_dir): os.makedirs(select_real_dir)
imgs_name = os.listdir(tamper_crop_path)
imgs_name.sort()
print(len(imgs_name))
# imgs_name = imgs_name[0*10000:1*10000]
for img_name in tqdm(imgs_name):
    print(img_name)
    img = io.imread(tamper_crop_path + img_name)
    mask_name = img_name
    mask = Image.open(mask_crop_path + mask_name).convert('L')
    mask = np.array(mask, dtype=np.uint8)
    if np.sum(mask) != 0:
        print(1)
        shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
    else:
        print(0)
        shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)


# tamper_crop_path = '/data1/zhengkengtao/findit/findit_4types/split_100_20_600/crop512_stride512/test_imgs_crop512stride512/'
# mask_crop_path = '/data1/zhengkengtao/findit/findit_4types/split_100_20_600/crop512_stride512/test_gt3_crop512stride512/'
# select_tamper_dir = '/data1/zhengkengtao/findit/findit_4types/split_100_20_600/crop512_stride512/DIDNetcrop/test_tamper/'
# select_real_dir = '/data1/zhengkengtao/findit/findit_4types/split_100_20_600/crop512_stride512/DIDNetcrop/test_real/'
# if not os.path.exists(select_tamper_dir): os.makedirs(select_tamper_dir)
# if not os.path.exists(select_real_dir): os.makedirs(select_real_dir)
# imgs_name = os.listdir(tamper_crop_path)
# imgs_name.sort()
# print(len(imgs_name))
# # imgs_name = imgs_name[0*10000:1*10000]
# for img_name in tqdm(imgs_name):
#     print(img_name)
#     img = io.imread(tamper_crop_path + img_name)
#     mask_name = 'gt3_' + img_name
#     mask = Image.open(mask_crop_path + mask_name).convert('L')
#     mask = np.array(mask, dtype=np.uint8)
#     mask[mask == 255] = 0
#     mask[mask == 76] = 1
#     mask[mask == 29] = 0
#     if np.sum(mask) != 0:
#         print(1)
#         shutil.copyfile(tamper_crop_path + img_name, select_tamper_dir + img_name)
#     else:
#         print(0)
#         shutil.copyfile(tamper_crop_path + img_name, select_real_dir + img_name)
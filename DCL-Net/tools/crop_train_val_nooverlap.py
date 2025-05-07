import os
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import warnings
import cv2
from tqdm import tqdm
from PIL import Image, ImageFile
# scikit-image读取和存储格式是RGB，也是numpy.ndarray格式
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------裁剪-----------------------------------------------------------------------------------------------
tamper_path = '/pubdata/zhengkengtao/docimg/printer/1_2/train_imgs/'
mask_path = '/pubdata/zhengkengtao/docimg/printer/1_2/train_gt3/'
print(len(os.listdir(mask_path)))
d = 512 # 裁剪的图像大小
tamper_crop_path = '/pubdata/zhengkengtao/docimg/printer/1_2/train_imgs_crop512stride512/'
mask_crop_path = '/pubdata/zhengkengtao/docimg/printer/1_2/train_gt3_crop512stride512/'
if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# tamper_crop_names = os.listdir(tamper_crop_path)
# print(len(tamper_crop_names))
imgs_name = os.listdir(tamper_path)
imgs_name.sort()
imgs_name = imgs_name[0*180:1*180]
file_path = '/pubdata/zhengkengtao/docimg/printer/1_2/crop512_train_log.txt'.format(d, d)
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
        mask_name = img_name.replace('psc', 'gt3', 1)
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


# # ----------裁剪右边下面剩余部分-----------------------------------------------------------------------------------------------
# tamper_path = '/pubdata/zhengkengtao/docimg/docimg_split811/train_images/'
# mask_path = '/pubdata/zhengkengtao/docimg/docimg_split811/train_gt3/'
# print(len(os.listdir(mask_path)))
# d = 512 # 裁剪的图像大小
# tamper_crop_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop{}x{}/trainextra_images/'.format(d, d)
# mask_crop_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop{}x{}/trainextra_gt3/'.format(d, d)
# if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
# if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# # tamper_crop_names = os.listdir(tamper_crop_path)
# # print(len(tamper_crop_names))
# imgs_name = os.listdir(tamper_path)
# imgs_name.sort()
# imgs_name = imgs_name[0*400: 1*400]
# file_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop{}x{}/crop_train_extra_log.txt'.format(d, d)
# with open(file_path, 'r') as f:
#     for line in f:
#         cropped_img = line.strip('\n')
#         imgs_name.remove(cropped_img)
# # 'w'代表着每次运行都覆盖内容，'a'代表着追加内容
# with open(file_path, 'a') as f:
#     for img_name in tqdm(imgs_name):
#         # print(img_name)
#         img = io.imread(tamper_path + img_name)
#         mask_name = img_name.replace('psc', 'gt3', 1)
#         mask = io.imread(mask_path + mask_name)
#         h, w = img.shape[0], img.shape[1]
#         h_num, w_num = h // d, w // d
#         hleft, wleft = h % d, w % d
#         if hleft == 0 and wleft != 0:
#             print('case1:', hleft, wleft)
#             for i in range(h_num):
#                 tamper_block = img[i * d:(i + 1) * d, w-d:, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, w_num), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[i * d:(i + 1) * d, w - d:]
#                 else:
#                     mask_block = mask[i * d:(i + 1) * d, w - d:, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, w_num), mask_block)
#         if hleft != 0 and wleft == 0:
#             print('case2:', hleft, wleft)
#             for j in range(w_num):
#                 tamper_block = img[h-d:, j * d:(j + 1) * d, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (h_num, j), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[h-d:, j * d:(j + 1) * d]
#                 else:
#                     mask_block = mask[h-d:, j * d:(j + 1) * d, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (h_num, j), mask_block)
#         if hleft != 0 and wleft != 0:
#             print('case3:', hleft, wleft)
#             for i in range(h_num):
#                 tamper_block = img[i * d:(i + 1) * d, w-d:, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, w_num), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[i * d:(i + 1) * d, w - d:]
#                 else:
#                     mask_block = mask[i * d:(i + 1) * d, w - d:, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, w_num), mask_block)
#             for j in range(w_num):
#                 tamper_block = img[h-d:, j * d:(j + 1) * d, :]
#                 tamper_block = np.array(tamper_block, dtype=np.uint8)
#                 io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (h_num, j), tamper_block)
#                 if len(mask.shape) == 2:
#                     mask_block = mask[h-d:, j * d:(j + 1) * d]
#                 else:
#                     mask_block = mask[h-d:, j * d:(j + 1) * d, :]
#                 mask_block = np.array(mask_block, dtype=np.uint8)
#                 io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (h_num, j), mask_block)
#             tamper_block = img[h - d:, w-d:, :]
#             tamper_block = np.array(tamper_block, dtype=np.uint8)
#             io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (h_num, w_num), tamper_block)
#             if len(mask.shape) == 2:
#                 mask_block = mask[h - d:, w-d:]
#             else:
#                 mask_block = mask[h - d:, w-d:, :]
#             mask_block = np.array(mask_block, dtype=np.uint8)
#             io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (h_num, w_num), mask_block)
#         f.write(img_name + '\n')
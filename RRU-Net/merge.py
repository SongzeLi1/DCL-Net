## Merging the predicted patches
import warnings
from PIL import ImageFile, Image
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import cv2
import time
import numpy as np


def get_row(block_name):
    the_row = block_name.split("_", -1)[-2]
    return the_row
def get_col(block_name):
    ending = block_name.split("_", -1)[-1]
    the_col = ending.split(".", -1)[0]
    return the_col
def patch_concat(img_path, patch_list, m1_all, m2_all, n1_all, n2_all):
    img_cv = cv2.imread(img_path)
    [M, N, C] = img_cv.shape
    img_save = np.zeros([M, N, C])
    time_save = np.zeros([M, N])
    len = m1_all.shape[0]
    for i in range(len):
        patch = patch_list[i]
        m1 = m1_all[i]
        m2 = m2_all[i]
        n1 = n1_all[i]
        n2 = n2_all[i]
        img_save[m1:m2, n1:n2] = img_save[m1:m2, n1:n2] + patch
        time_save[m1:m2, n1:n2] = time_save[m1:m2, n1:n2] + 1
    img = np.divide(img_save, time_save)
    return img


if __name__ == '__main__':

    image_dataset_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_images/'
    # dst_dir = '/pubdata/zhengkengtao/exps/0711_UConvNeXtSE_docimg_split811_noAug/1203_DIFNet_docimgsplit811_crop512train/difnet_512x512/test_docimgsplit811crop512stride256_bestckptepoch162/'
    # image_dataset_path = '/data1/zhengkengtao/SUPATLANTIQUE/tamper/'
    dst_dir = '/pubdata/zhengkengtao/exps/1023_MVSSNet_docimg_split811_png_OrigMosaicOneKind/test_docimgsplit811test_bestepoch19_crop512stride256/pred_512x512/'

    mapblock_read_path = dst_dir + 'unthreshold/'
    mapmerge_write_path = dst_dir + 'mapblockmerge/'
    threshold = 0.5
    merge_write_path = dst_dir + 'predblockmerge_thres{}/'.format(threshold)

    if not os.path.exists(merge_write_path): os.makedirs(merge_write_path)
    if not os.path.exists(mapmerge_write_path): os.makedirs(mapmerge_write_path)
    image_name_list = os.listdir(image_dataset_path)
    mapblock_name_list = os.listdir(mapblock_read_path)

    block_size = 512
    step = 256
    image_name_list.sort()
    for image_name in image_name_list:
        print(image_name)
        t1 = time.time()
        image = cv2.imread(image_dataset_path + image_name, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[0], image.shape[1]
        mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] in mapblock] # for DOC DID SUPAT
        # mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] == mapblock.split("_", -1)[0]] # for Ali
        mapblock_name.sort()

        # ---合并概率图---
        mapmerge_block = np.zeros((height, width))
        maptimes_save = np.zeros([height, width])
        for mapblock in mapblock_name:
            mappredict_block = Image.open(mapblock_read_path + mapblock).convert('L')
            mappredict_block = np.array(mappredict_block, dtype=np.uint8)
            mappredict_block = mappredict_block
            the_row = get_row(mapblock)
            the_col = get_col(mapblock)
            right = block_size + int(the_col) * step
            left = right - block_size
            bottom = block_size + int(the_row) * step
            top = bottom - block_size
            if bottom <= height and right > width:
                y1, y2, x1, x2 = top, bottom, width - block_size, width
            elif bottom > height and right <= width:
                y1, y2, x1, x2 = height - block_size, height, left, right
            elif bottom > height and right > width:
                y1, y2, x1, x2 = height - block_size, height, width - block_size, width
            else:
                y1, y2, x1, x2 = top, bottom, left, right
            mapmerge_block[y1:y2, x1:x2] += mappredict_block
            maptimes_save[y1:y2, x1:x2] += 1
        mapmerge_block = mapmerge_block / maptimes_save
        mapmerge_block_ = Image.fromarray(np.uint8(mapmerge_block))
        mapmerge_block_ = mapmerge_block_.convert('L')
        mapmerge_block_.save(mapmerge_write_path + image_name[:-4] + '.png')

        # ---合并二值图---
        pred = mapmerge_block.copy()
        # print(int(threshold * 255))
        pred[pred <= int(threshold * 255)] = 0
        pred[pred > int(threshold * 255)] = 255
        pred = Image.fromarray(np.uint8(pred))
        pred = pred.convert('L')
        pred.save(merge_write_path + image_name[:-4] + '.png')

        print('merge one img time:', time.time() - t1)
    print("finish")

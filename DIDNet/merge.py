## Merging the predicted patches
import warnings
from PIL import ImageFile
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import os
import cv2
def get_row(block_name):
    the_row = block_name.split("_", -1)[-2]
    return the_row
def get_col(block_name):
    ending = block_name.split("_", -1)[-1]
    the_col = ending.split(".", -1)[0]
    return the_col
def merge_predict_mask():
    # image_dataset_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
    # block_read_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86/'
    # merge_write_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/blockmerge/'
    # mapblock_read_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86_map/'
    # mapmerge_write_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86_mapblockmerge/'

    # image_dataset_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_imgs/'
    # block_read_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_epoch10/test_save/'
    # merge_write_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_epoch10/blockmerge/'
    # mapblock_read_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_epoch10_map/'
    # mapmerge_write_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_epoch10_mapblockmerge/'

    # image_dataset_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/img/'
    # block_read_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/block/'
    # merge_write_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/blockmerge/'
    # mapblock_read_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/mapblock/'
    # mapmerge_write_path = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/mapblockmerge/'

    # image_dataset_path = '/data1/zhengkengtao/docimg/tamper_mosaic/'
    # block_read_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/block/'
    # merge_write_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/blockmerge/'
    # mapblock_read_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/mapblock/'
    # mapmerge_write_path = '/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/mapblockmerge/'

    image_dataset_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
    block_read_path = '/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/block/'
    merge_write_path = '/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/blockmerge/'
    mapblock_read_path = '/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/mapblock/'
    mapmerge_write_path = '/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/mapblockmerge/'

    if not os.path.exists(merge_write_path): os.makedirs(merge_write_path)
    if not os.path.exists(mapmerge_write_path): os.makedirs(mapmerge_write_path)
    image_name_list = os.listdir(image_dataset_path)
    block_name_list = os.listdir(block_read_path)
    mapblock_name_list = os.listdir(mapblock_read_path)
    # print(len(block_name_list))
    block_size = 64
    step = 32
    image_name_list.sort()

    print(len(image_name_list))
    mergered_imgs = os.listdir(merge_write_path)
    for i in image_name_list:
        if i in mergered_imgs:
            image_name_list.remove(i)
    print(len(image_name_list))

    for image_name in image_name_list:
        image = cv2.imread(image_dataset_path + image_name, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[0], image.shape[1]

        block_name = [block for block in block_name_list if image_name.split(".", -1)[0] in block] # for DOC DID
        mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] in mapblock] # for DOC DID

        # block_name = [block for block in block_name_list if image_name.split(".", -1)[0] == block.split("_", -1)[0]] # for Ali
        # mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] == mapblock.split("_", -1)[0]] # for Ali

        block_name.sort()
        mapblock_name.sort()
        print(image_name)

        # ---合并二值图---
        merge_block = np.zeros((height, width))
        for block in block_name:
            predict_block = cv2.imread(block_read_path + block, cv2.IMREAD_UNCHANGED)
            predict_block = predict_block / 255
            the_row = get_row(block)
            the_col = get_col(block)
            # print(the_row, the_col, predict_block.shape)
            right = block_size + int(the_col) * step
            left = right - block_size
            bottom = block_size + int(the_row) * step
            top = bottom - block_size
            # print(bottom-top, right-left)
            # merge_block[top:bottom, left:right] += predict_block
            # t = merge_block[top:bottom, left:right]
            # print(top, bottom, left, right, t.shape)
            if bottom <= height and right > width:
                merge_block[top:bottom, width-block_size:width] +=  predict_block
            elif bottom > height and right <= width:
                merge_block[height-block_size:height, left:right] +=  predict_block
            elif bottom > height and right > width:
                merge_block[height-block_size:height, width-block_size:width] +=  predict_block
            else:
                merge_block[top:bottom, left:right] += predict_block
        merge_block[merge_block > 3] = 255
        merge_block[merge_block < 4] = 0
        cv2.imwrite(merge_write_path + image_name[:-4] + '.png', merge_block)

        # ---合并概率图---
        mapmerge_block = np.zeros((height, width))
        for mapblock in mapblock_name:
            mappredict_block = cv2.imread(mapblock_read_path + mapblock, cv2.IMREAD_UNCHANGED)
            mappredict_block = mappredict_block / 255
            the_row = get_row(mapblock)
            the_col = get_col(mapblock)
            # print(the_row, the_col, predict_block.shape)
            right = block_size + int(the_col) * step
            left = right - block_size
            bottom = block_size + int(the_row) * step
            top = bottom - block_size
            # print(bottom-top, right-left)
            # merge_block[top:bottom, left:right] += predict_block
            # t = merge_block[top:bottom, left:right]
            # print(top, bottom, left, right, t.shape)
            if bottom <= height and right > width:
                mapmerge_block[top:bottom, width - block_size:width] += mappredict_block
            elif bottom > height and right <= width:
                mapmerge_block[height - block_size:height, left:right] += mappredict_block
            elif bottom > height and right > width:
                mapmerge_block[height - block_size:height, width - block_size:width] += mappredict_block
            else:
                mapmerge_block[top:bottom, left:right] += mappredict_block
        mapmerge_block = mapmerge_block / 4
        # print(mapmerge_block.min(), mapmerge_block.max())
        cv2.imwrite(mapmerge_write_path + image_name[:-4] + '.png', np.uint8(mapmerge_block * 255))
    print("finish")

merge_predict_mask()
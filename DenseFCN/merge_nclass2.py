import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from PIL import Image, ImageFile
import warnings
import time
import numpy as np
import cv2
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_row(block_name):
    the_row = block_name.split("_", -1)[-2]
    return the_row

def get_col(block_name):
    ending = block_name.split("_", -1)[-1]
    the_col = ending.split(".", -1)[0]
    return the_col


if __name__=="__main__":
    # self.blocks = nn.ModuleList(self.blocks) 取消

    warnings.filterwarnings('ignore')
    n_class = 3
    mask_threshold, viz, save_map, save_mask, cal_metrics, batch_size = 0.5, True, True, True, False, 1
    input_size = [512, 512] # None [1920, 1920] [1792, 1792] [1664, 1664] [1440, 1440] [768, 768]
    # -------------
    image_dataset_path = '/data1/zhengkengtao/docimg/tamper_mosaic/'
    gt_path = '/data1/zhengkengtao/docimg/gt3/'
    test_img_path = '/data1/zhengkengtao/docimg/tamper_mosaic_crop512stride256/'
    test_gt_path = '/data1/zhengkengtao/docimg/gt3_crop512stride256/'
    save_results_path = '/data1/zhengkengtao/exps/0717_denseFCN_Alinew_trainsplit811_crop512x512/testdocimgall_crop512stride256/'
    # --------------
    map_path = save_results_path + 'unthreshold/'
    pred_path = save_results_path + 'threshold_0.5/'


    # ----------------merge----------------------
    dst_dir = save_results_path

    mapblock_read_path = map_path
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
    image_name_list.remove('psc_honor60pro_s4p_279_add.png')
    print(len(image_name_list))
    for image_name in image_name_list:
        print(image_name)
        t1 = time.time()
        image = cv2.imread(image_dataset_path + image_name, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[0], image.shape[1]
        mapblock_name = [mapblock for mapblock in mapblock_name_list if
                         image_name.split(".", -1)[0] in mapblock]  # for DOC DID SUPAT
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



from difutils.metric import *
from PIL import Image, ImageFile
from difutils.utils import inial_logger
import os
import csv
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
# 输出一通道——原始脱敏视为一类计算指标
gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
# gt_path = '/data1/zhengkengtao/docimg/gt3/'
testresult_dir = save_results_path
# testresult_dir = '/data1/zhengkengtao/exps/0609_denseFCN_docimg_split811_crop512x512selectnoblank/valf10.8023_test/'
map_path = mapmerge_write_path
pred_path = merge_write_path

save_metrics_path = testresult_dir
print(len(os.listdir(pred_path)))
logger = inial_logger(os.path.join(save_metrics_path, 'origmosaic_samekind_metric.log'))
metrics_csv = save_metrics_path +  "origmosaic_samekind_metric.csv"
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['img','gt','f1','iou','mcc','auc','precision','tpr_recall','tnr','fpr','fnr'])
f.close()
filenames = os.listdir(pred_path)
filenames.sort()
f1s, ious, mccs, aucs, precisions, tprs, tnrs, fprs, fnrs= [], [], [], [], [], [], [], [], []
for i in range(len(filenames)):
    name = filenames[i]

    # map = np.load(map_path + '{}.npy'.format(name[:-4]))
    # # print(map[100,100,0].dtype)
    # map = np.float16(map)
    # # print(map[100,100,0],map[100,100,1],map[100,100,2])
    map = Image.open(map_path + name)
    map = np.array(map) / 255
    # # print(map.max(),map.min())

    pred = Image.open(pred_path + name).convert('L')
    pred = np.array(pred)

    pred[pred == 255] = 1

    # # docimg
    gt_name = name.replace('psc_', 'gt3_')
    gt = Image.open(gt_path + gt_name).convert('L')
    gt = np.array(gt)
    gt[gt == 255] = 0
    gt[gt == 76] = 1
    gt[gt == 29] = 0

    # # Alinew
    # gt_name = name
    # gt = Image.open(gt_path + gt_name).convert('L')
    # gt = np.array(gt)
    # gt[gt != 0] = 1

    tpr_recall, tnr, fpr, fnr, precision, f1, mcc, iou, tn, tp, fn, fp, auc = get_metrics(gt, pred, map)
    logger.info('{}, {}, f1:{:.4f}, iou:{:.4f}, mcc:{:.4f}, fpr:{:.4f}, auc:{:.4f}'.format(i, name, f1, iou, mcc, fpr, auc))
    logger.info('======================================================================================================')
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow([name, gt_name, f1, iou, mcc, auc, precision, tpr_recall, tnr, fpr, fnr])
    f.close()
    f1s.append(f1)
    ious.append(iou)
    mccs.append(mcc)
    aucs.append(auc)
    precisions.append(precision)
    tprs.append(tpr_recall)
    tnrs.append(tnr)
    fprs.append(fpr)
    fnrs.append(fnr)

f1_mean, iou_mean, mcc_mean, auc_mean, precision_mean, tpr_mean, tnr_mean, fpr_mean, fnr_mean = \
np.mean(f1s), np.mean(ious), np.mean(mccs), np.mean(aucs), np.mean(precisions), np.mean(tprs), np.mean(tnrs), np.mean(fprs), np.mean(fnrs)
logger.info('======================================================================================================')
logger.info('test num:{}, f1:{:.4f}, iou:{:.4f}, mcc:{:.4f}, fpr:{:.4f}, auc:{:.4f}'.format(len(filenames), f1_mean, iou_mean, mcc_mean, fpr_mean, auc_mean))
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['Average', len(filenames), f1_mean, iou_mean, mcc_mean, auc_mean, precision_mean, tpr_mean, tnr_mean, fpr_mean, fnr_mean])
csv_writer.writerow(['END','END','f1','iou','mcc','auc','precision','tpr_recall','tnr','fpr','fnr'])
f.close()



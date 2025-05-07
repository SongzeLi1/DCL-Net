import os
from utils.metric import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import cv2
from utils.utils import inial_logger
from utils.segmetric import *
import time
import os
import csv
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 输出一通道——篡改脱敏视为一类计算指标
# gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
testresult_dir = '/data1/zhengkengtao/exps/1615_DIDNet_Alinew_trainsplit1000_200_2800_noAug_pretrainwithdocimgsplit811Aug/test_Alinew2800_epoch3/'
map_path = testresult_dir + 'mapblockmerge/'
pred_path = testresult_dir +  'blockmerge/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
# pred_path = '/data1/zhengkengtao/exps/0727_Noiseprint_Alinew_trainall/out_mask_thres0.5/'
# gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# pred_path = '/data1/zhengkengtao/exps/0724_denseFCN_Alinew_trainsplit19_pretrainwithdocimgsplit811_crop512x512/0727_test_Alinewtrainsplit19_test_valf10.142758/thresholded_0.5/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
# pred_path = '/data1/zhengkengtao/exps/0723_MVSSNet_docimg_split811_png/test_epoch63/pred_1440x1440/threshold_0.5/'
save_metrics_path = testresult_dir
print(len(os.listdir(pred_path)))
logger = inial_logger(os.path.join(save_metrics_path, 'forgerymosaic_samekind_metric.log'))
metrics_csv = save_metrics_path +  "forgerymosaic_samekind_metric.csv"
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

    # # # docimg
    # gt_name = name.replace('psc_', 'gt3_')
    # gt = Image.open(gt_path + gt_name).convert('L')
    # gt = np.array(gt)
    # gt[gt == 255] = 0
    # gt[gt == 76] = 1
    # gt[gt == 29] = 1

    # Alinew
    gt_name = name
    gt = Image.open(gt_path + gt_name).convert('L')
    gt = np.array(gt)
    gt[gt != 0] = 1

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



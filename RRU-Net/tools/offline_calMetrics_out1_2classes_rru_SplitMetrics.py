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


# 其它算法训练时：篡改和脱敏一类
# 算其它算法指标 ———— 篡改脱敏视为两类（这里计算指标有点逻辑bug，默认其它算法也能够区分脱敏和篡改，同时虚警的地方认为是篡改）
gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
testresult_dir = '/data1/zhengkengtao/exps/0716_RRUNet_docimg_split811png_crop512noblank_noAug/rru_512x512/test_docimgsplit811test_bestepoch201/'
map_path = testresult_dir + 'mapblockmerge/'
pred_path = testresult_dir +  'predblockmerge_thres0.5/'
save_metrics_path = testresult_dir
print(len(os.listdir(pred_path)))
metrics_csv = save_metrics_path +  "forgerymosaic_samekind_SplitMetrics.csv"
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['img','gt','f1-1','iou-1','mcc-1','auc-1','fpr-1','f1-2','iou-2','mcc-2','auc-2','fpr-2'])
f.close()
filenames = os.listdir(pred_path)
filenames.sort()
f1forgerys, iouforgerys, mccforgerys, aucforgerys, fprforgerys, f1mosaics, ioumosaics, mccmosaics, aucmosaics, fprmosaics= [], [], [], [], [], [], [], [], [], []
for i in range(len(filenames)):
    name = filenames[i]

    map = Image.open(map_path + name)
    map = np.array(map) / 255

    pred = Image.open(pred_path + name).convert('L')
    pred = np.array(pred)
    pred[pred == 255] = 1

    gt_name = name.replace('psc_', 'gt3_')
    gt = Image.open(gt_path + gt_name).convert('L')
    gt = np.array(gt)
    gt[gt == 255] = 0
    gt[gt == 76] = 1
    gt[gt == 29] = 2
    gtforgery, gtmosaic = gt.copy(), gt.copy()
    gtforgery[gtforgery == 2] = 0
    gtmosaic[gtmosaic == 1] = 0
    gtmosaic[gtmosaic == 2] = 1

    predforgery, predall = pred.copy(), pred.copy()
    predall[predall != 0] = 1
    predforgery[gt == 2] = 0
    predmosaic = predall - predforgery
    tpr_recall1, tnr1, fpr1, fnr1, precision1, f11, mcc1, iou1, tn1, tp1, fn1, fp1, auc1 = get_metrics(gtforgery, predforgery, map)
    tpr_recall2, tnr2, fpr2, fnr2, precision2, f12, mcc2, iou2, tn2, tp2, fn2, fp2, auc2 = get_metrics(gtmosaic, predmosaic, map)

    print('{}, {}, f1forgery:{:.3f}, iouforgery:{:.3f}, mccforgery:{:.3f}, aucforegry:{:.3f}, fprforgery:{:.3f}'
                'f1mosaic:{:.3f}, ioumosaic:{:.3f}, mccmosaic:{:.3f}, aucmosaic:{:.3f}, fprmosaic:{:.3f}'
                .format(i, name, f11, iou1, mcc1, auc1, fpr1, f12, iou2, mcc2, auc2, fpr2))
    print('======================================================================================================')
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow([name, gt_name, f11, iou1, mcc1, auc1, fpr1, f12, iou2, mcc2, auc2, fpr2])
    f.close()
    f1forgerys.append(f11)
    iouforgerys.append(iou1)
    mccforgerys.append(mcc1)
    aucforgerys.append(auc1)
    fprforgerys.append(fpr1)
    f1mosaics.append(f12)
    ioumosaics.append(iou2)
    mccmosaics.append(mcc2)
    aucmosaics.append(auc2)
    fprmosaics.append(fpr2)

f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean = np.mean(f1forgerys), np.mean(iouforgerys), np.mean(mccforgerys), np.mean(aucforgerys), np.mean(fprforgerys)
f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean= np.mean(f1mosaics), np.mean(ioumosaics), np.mean(mccmosaics), np.mean(aucmosaics), np.mean(fprmosaics)
print('======================================================================================================')
print('test num:{}, f1foregry:{:.3f}, iouforgey:{:.3f}, mccforgery:{:.3f}, aucforgery:{:.3f}, fprforgery:{:.3f}, '
      'f1mosaic:{:.3f}, ioumosaic:{:.3f}, mccmosaic:{:.3f}, aucmosaic:{:.3f}, fprmosaic:{:.3f}'
            .format(len(filenames), f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean,
                    f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean))
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['END','END','f1-1','iou-1','mcc-1','auc-1','fpr-1','f1-2','iou-2','mcc-2','auc-2','fpr-2'])
csv_writer.writerow(['Average', len(filenames), f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean,
                     f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean])
f.close()
import os
from utils.metric import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import cv2
from utils.utils import inial_logger
from utils.segmetric import *
import csv
import time
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 输出三通道——篡改脱敏视为两类
gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_gt3/'
testresult_dir = '/pubdata/zhengkengtao/exps/1509_DIFNetV2_Alinewtrainsplit415_ft_Aug_ftwith_docimgsplit811AugNoise/difnetv2_768x768/test_Alinewsplit415test_bestepoch240/'
map_path = testresult_dir + 'mapblockmerge/'
pred_path = testresult_dir +  'predblockmerge/'
save_metrics_path = testresult_dir
metrics_csv = save_metrics_path +  "3kinds_metric.csv"
logger = inial_logger(os.path.join(save_metrics_path, '3kinds_metric.log'))
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['img','gt','f1-1','iou-1','mcc-1','auc-1','fpr-1','f1-2','iou-2','mcc-2','auc-2','fpr-2','miou','mauc','pa','mpa'])
f.close()
filenames = os.listdir(pred_path)
filenames.sort()
mious, maucs, pas, mpas = [], [], [], []
f1forgerys, iouforgerys, mccforgerys, aucforgerys, fprforgerys= [], [], [], [], []
f1mosaics, ioumosaics, mccmosaics, auc_mosaics, fprmosaics = [], [], [], [], []
for i in range(len(filenames)):
    t1 = time.time()
    name = filenames[i]

    # map = np.load(map_path + '{}.npy'.format(name[:-4]))
    map = Image.open(map_path + name)
    map = np.array(map) / 255
    # print(map[100, 100, 0] + map[100, 100, 1] + map[100, 100, 2])
    map_forgery = map[:, :, 1]
    map_mosaic = map[:, :, 2]

    pred = Image.open(pred_path + name).convert('L')
    pred = np.array(pred)
    pred[pred == 255] = 0
    pred[pred == 76] = 1
    pred[pred == 29] = 2
    gt_name = name.replace('psc_', 'gt3_')
    gt_name = gt_name.replace('mosaic_', 'c_')
    gt_name = gt_name.replace('pngps', 'pngms')
    gt_name = gt_name.replace('forgery', 'gt')
    gt = Image.open(gt_path + gt_name).convert('L')
    gt = np.array(gt, dtype=np.uint8)
    # docimg
    gt[gt == 255] = 0
    gt[gt == 76] = 1
    gt[gt == 29] = 2
    # # Alinew supatlantique certificate
    # gt[gt != 0] = 1
    # # findit-mosaic、supatlantique-mosaic
    # gt[gt !=0 ] = 2

    metric = SegmentationMetric(3)  # 2表示有2个分类，有几个分类就填几
    # print(pred.shape, gt.shape)
    hist = metric.addBatch(pred, gt)  # [h,w], [h,w]
    # IoU = metric.IntersectionOverUnion() # 每一个
    mIoU = metric.meanIntersectionOverUnion()
    pa = metric.pixelAccuracy()
    # cpa = metric.classPixelAccuracy() # 每一个
    mpa = metric.meanPixelAccuracy() # CPA的平均
    # mauc = get_multiclass_mean_auc(gt, map) # 相加不为1算不了
    mauc = 0
    mious.append(mIoU)
    pas.append(pa)
    mpas.append(mpa)
    maucs.append(mauc)

    gtforgery, gtmosaic = gt.copy(), gt.copy()
    gtforgery[gtforgery == 2] = 0
    gtmosaic[gtmosaic == 1] = 0
    gtmosaic[gtmosaic == 2] = 1
    predforgery, predmosaic = pred.copy(), pred.copy()
    predforgery[predforgery == 2] = 0
    predmosaic[predmosaic == 1] = 0
    predmosaic[predmosaic == 2] = 1
    f11, iou1, mcc1, auc1, fpr1 = get_f1_iou_mcc_auc_fpr(gtforgery, predforgery, map_forgery)
    f12, iou2, mcc2, auc2, fpr2 = get_f1_iou_mcc_auc_fpr(gtmosaic, predmosaic, map_mosaic)

    logger.info('{}, {}, f1forgery:{:.4f}, f1mosaic:{:.4f}'.format(i, name, f11, f12))
    logger.info('======================================================================================================')
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow([name, gt_name, f11, iou1, mcc1, auc1, fpr1, f12, iou2, mcc2, auc2, fpr2, mIoU, mauc, pa, mpa])
    f.close()

    f1forgerys.append(f11)
    iouforgerys.append(iou1)
    mccforgerys.append(mcc1)
    aucforgerys.append(auc1)
    fprforgerys.append(fpr1)
    f1mosaics.append(f12)
    ioumosaics.append(iou2)
    mccmosaics.append(mcc2)
    auc_mosaics.append(auc2)
    fprmosaics.append(fpr2)

    # print(time.time()-t1)
miou_mean, mauc_mean, pa_mean, mpa_mean = np.mean(mious), np.mean(maucs), np.mean(pas), np.mean(mpas)
f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean = np.mean(f1forgerys), np.mean(iouforgerys), np.mean(mccforgerys), np.mean(aucforgerys), np.mean(fprforgerys)
f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean = np.mean(f1mosaics), np.mean(ioumosaics), np.mean(mccmosaics), np.mean(auc_mosaics), np.mean(fprmosaics)
logger.info('======================================================================================================')
logger.info('test num:{}, f1foregry:{:.4f}, aucforgery:{:.4f}, fprforgery:{:.4f}, f1mosaic:{:.4f}, aucmosaic:{:.4f}, fprmosaic:{:.4f}'
            .format(len(filenames), f1forgery_mean, aucforgery_mean, fprforgery_mean, f1mosaic_mean, aucmosaic_mean, fprmosaic_mean))
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['Average', len(filenames),
                     f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean,
                     f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean,
                     miou_mean, mauc_mean, pa_mean, mpa_mean])
csv_writer.writerow(['END','END','f1-1','iou-1','mcc-1','auc-1','fpr-1','f1-2','iou-2','mcc-2','auc-2','fpr-2','miou','mauc','pa','mpa'])
f.close()


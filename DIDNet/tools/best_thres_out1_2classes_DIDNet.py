from utils.metric import *
from PIL import Image, ImageFile
from utils.utils import inial_logger
import os
import csv
import numpy as np
from sklearn.metrics import precision_recall_curve
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 输出一通道——篡改脱敏视为一类计算指标
# gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
# gt_path = '/data1/zhengkengtao/docimg/gt3/'
# testresult_dir = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/'
# testresult_dir = '/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/'
testresult_dir = '/data1/zhengkengtao/exps/0728_DIDNet_Alinewtrain_split19_pretrainwithdocimgsplit811/test_epoch5/'
# map_path = testresult_dir + 'test_epoch86_mapblockmerge/'
# pred_path = testresult_dir + 'test_epoch86_blockmerge/'
map_path = testresult_dir + 'mapblockmerge/'
pred_path = testresult_dir + 'blockmerge/'
save_bestthres_mask = True
bestthres_pred_path = testresult_dir + 'bestthreshold/'
if (os.path.exists(bestthres_pred_path) == False): os.makedirs(bestthres_pred_path)
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
# pred_path = '/data1/zhengkengtao/exps/0727_Noiseprint_Alinew_trainall/out_mask_thres0.5/'
# gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# pred_path = '/data1/zhengkengtao/exps/0724_denseFCN_Alinew_trainsplit19_pretrainwithdocimgsplit811_crop512x512/0727_test_Alinewtrainsplit19_test_valf10.142758/thresholded_0.5/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
# pred_path = '/data1/zhengkengtao/exps/0723_MVSSNet_docimg_split811_png/test_epoch63/pred_1440x1440/threshold_0.5/'
save_metrics_path = testresult_dir
print(len(os.listdir(map_path)))
metrics_csv = save_metrics_path +  "bestthres_forgerymosaic_samekind_metric.csv"
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['img','bestthres','f1','iou','mcc','auc','precision','tpr_recall','tnr','fpr','fnr'])
f.close()
filenames = os.listdir(map_path)
filenames.sort()
f1s, ious, mccs, aucs, precisions, tprs, tnrs, fprs, fnrs= [], [], [], [], [], [], [], [], []
for i in range(len(filenames)):
    name = filenames[i]
    print(name)
    map = Image.open(map_path + name)
    map = np.array(map) / 255
    # # print(map.max(),map.min())

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

    # 计算最优阈值
    precision, recall, thresholds = precision_recall_curve(gt.flatten().astype(int), map.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_threshold = thresholds[np.argmax(f1)]
    # print(map.max(), map.min())
    pred = (map > best_threshold).astype(np.uint8)

    # 比较最优阈值与0.5阈值的f1
    bestthresf1 = f1_score(gt, pred)
    # print(map.max(), map.min())
    halfthrespred = Image.open(pred_path + name).convert('L')
    halfthrespred = np.array(halfthrespred)
    halfthrespred[halfthrespred == 255] = 1
    halfthresf1 = f1_score(gt, halfthrespred)
    # print(halfthresf1)
    print(bestthresf1, halfthresf1)
    if halfthresf1 > bestthresf1:
        best_threshold = 0.5
        pred = halfthrespred
    print(best_threshold)
    if save_bestthres_mask:
        pred_ = Image.fromarray(np.uint8(pred * 255))
        pred_ = pred_.convert('L')
        pred_.save(bestthres_pred_path + name[:-4] + '.png')

    tpr_recall, tnr, fpr, fnr, precision, f1, mcc, iou, tn, tp, fn, fp, auc = get_metrics(gt, pred, map)
    print('{}, {}, bestthres:{:.3f} f1:{:.3f}, iou:{:.3f}, mcc:{:.3f}, fpr:{:.3f}, auc:{:.3f}'.format(i, name, best_threshold, f1, iou, mcc, fpr, auc))
    print('======================================================================================================')
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow([name, best_threshold, f1, iou, mcc, auc, precision, tpr_recall, tnr, fpr, fnr])
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
print('======================================================================================================')
print('test num:{}, f1:{:.3f}, iou:{:.3f}, mcc:{:.3f}, fpr:{:.3f}, auc:{:.3f}'.format(len(filenames), f1_mean, iou_mean, mcc_mean, fpr_mean, auc_mean))
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['END','END','f1','iou','mcc','auc','precision','tpr_recall','tnr','fpr','fnr'])
csv_writer.writerow(['Average', len(filenames), f1_mean, iou_mean, mcc_mean, auc_mean, precision_mean, tpr_mean, tnr_mean, fpr_mean, fnr_mean])
f.close()

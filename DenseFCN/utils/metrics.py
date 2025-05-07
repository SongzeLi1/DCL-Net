import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def get_tn_tp_fn_fp(y_true, y_pred):
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred))).astype(np.float64)
    tp = np.sum(np.logical_and(               y_true ,                y_pred )).astype(np.float64)
    fn = np.sum(np.logical_and(               y_true , np.logical_not(y_pred))).astype(np.float64)
    fp = np.sum(np.logical_and(np.logical_not(y_true),                y_pred )).astype(np.float64)
    return tn, tp, fn, fp

def matthews_corrcoef(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if np.isnan(mcc):
        return 0.
    else:
        return mcc

def f1_score(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    f1 = 2*tp/(2*tp+fp+fn)
    if np.isnan(f1):
        return 0.
    else:
        return f1

def iou_measure(y_true, y_pred):
    _, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    return tp/(tp+fp+fn)

def get_metrics(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)

    f1 = 2*tp/(2*tp+fp+fn)
    if np.isnan(f1):
        f1 = 0.

    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if np.isnan(mcc):
        mcc = 0.

    iou = tp/(fp+tp+fn)
    if np.isnan(iou):
        iou = 0.
    tpr_recall = tp/(tp+fn)
    if np.isnan(tpr_recall):
        tpr_recall = 0.
    tnr = tn/(tn+fp)
    if np.isnan(tnr):
        tnr = 0.
    precision = tp/(tp+fp)
    if np.isnan(precision):
        precision = 0.

    return tpr_recall,tnr,precision,f1,mcc,iou,tn,tp,fn,fp


if __name__ == '__main__':
    # mask_path = '../testedImages/Ali/mask/'
    # pred_path = '../results/Ali/thresholded_0.5/'
    # save_metrics_path = '../results/Ali/metrics/'
    mask_path = '/data/zhengkengtao/NIST2016-508/mask/'
    pred_path = '/data/zhengkengtao/Dense-FCN/results/NIST2016-508/thresholded_0.5/'
    save_metrics_path = '/data/zhengkengtao/Dense-FCN/results/NIST2016-508/metrics/'
    # # 用Ali2104微调NIST训练出来的模型得到新模型，测试Ali601
    # mask_path = '/data/zhengkengtao/Ali/test/masks/'
    # pred_path = '/data/zhengkengtao/Dense-FCN/results/Ali_test_601/thresholded_0.5/'
    # save_metrics_path = '/data/zhengkengtao/Dense-FCN/results/Ali_test_601/metrics/'

    mask_names = os.listdir(mask_path)
    num = len(mask_names)
    print(num)
    f = open(os.path.join(save_metrics_path, 'metrics.txt'), 'w+')
    f1s = 0
    ious = 0
    mccs = 0
    for i in range(num):
        mask_name = mask_names[i]
        # 1.
        tmp = mask_name[2:]
        pred_name = 'PS' + tmp
        # 2.
        # pred_name = mask_name
        a_mask_path = mask_path + mask_name
        a_pred_path = pred_path + pred_name
        mask = mpimg.imread(a_mask_path)
        pred = mpimg.imread(a_pred_path)
        plt.show()
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(mask, cmap='gray')
        ax[1].imshow(pred, cmap='gray')
        tpr_recall, tnr, precision, f1, mcc, iou, tn, tp, fn, fp = get_metrics(mask, pred)
        # print(i, mask_name, pred_name, 'precision =', precision, 'f1 =', f1, 'iou =', iou, 'mcc =', mcc)
        print(i, 'precision =', precision, 'f1 =', f1, 'iou =', iou, 'mcc =', mcc)
        # 写入txt文件
        print(i, pred_name, 'tpr_recall =', tpr_recall, 'tnr =', tnr, 'precision =', precision, 'f1 =', f1, 'mcc =', mcc,
             'iou =', iou, 'tn =', tn, 'tp =', tp, 'fn =', fn, 'fp =', fp,  file=f)
        f1s = f1s + f1
        ious = ious + iou
        mccs = mccs + mcc
    f1_ave = f1s / num
    iou_ave = ious / num
    mcc_ave = mccs / num
    print('testedImages number:', num, 'f1:', f1_ave, 'iou:', iou_ave, 'mcc:', mcc_ave)
    print('------------------------------------------------------------------------------------------------------------', file=f)
    print('testedImages number:', num, 'f1:', f1_ave, 'iou:', iou_ave, 'mcc:', mcc_ave, file=f)
    f.close()
    print('finish compute metrics')

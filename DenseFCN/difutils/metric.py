import numpy as np
from sklearn import metrics


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc



# 其它指标 2022.3.3
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


def get_f1_iou(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    f1 = 2*tp/(2*tp+fp+fn)
    if np.isnan(f1):
        f1 = 0.
    iou = tp/(fp+tp+fn)
    if np.isnan(iou):
        iou = 0.
    return f1, iou


def get_metrics_without_auc(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)

    f1 = 2*tp/(2*tp+fp+fn)
    if np.isnan(f1):
        f1 = 0.

    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.00001)
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

    fpr = fp/(fp+tn) # 假阳性率/虚警率
    if np.isnan(fpr):
        fpr = 0.

    fnr = fn/(tp+fn) # 假阴性率
    if np.isnan(fnr):
        fnr = 0.

    precision = tp/(tp+fp+0.00001)
    if np.isnan(precision):
        precision = 0.

    return tpr_recall, tnr, fpr, fnr, precision, f1, mcc, iou, tn, tp, fn, fp


def get_metrics(y_true, y_pred, y_map):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)

    f1 = 2*tp/(2*tp+fp+fn)
    if np.isnan(f1):
        f1 = 0.

    mcc = (tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.00001)
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

    fpr = fp / (fp + tn)  # 假阳性率/虚警率
    if np.isnan(fpr):
        fpr = 0.

    fnr = fn / (tp + fn)  # 假阴性率
    if np.isnan(fnr):
        fnr = 0.

    precision = tp/(tp+fp+0.00001)
    if np.isnan(precision):
        precision = 0.

    auc = 0
    # 若一部分数据的标签全为0，即标签的类别只有一种，没有0和1两种，那么计算auc会报错
    try:
        auc = metrics.roc_auc_score(y_true.reshape(-1, ), y_map.reshape(-1, ))
        if np.isnan(auc):
            auc = 0.
    except ValueError:
        pass

    return tpr_recall, tnr, fpr, fnr, precision, f1, mcc, iou, tn, tp, fn, fp, auc


def get_multiclass_mean_auc(y_true, y_map):
    '''
        y_true_shape: (h,w) -> (h*w, ) 或 (h,w,c) -> (h*w, c)
        y_map_shape:  (h,w,c)-> (h*w, c)
        如：
        y_true: [[1,2],[0,2]] -> [0,1,0,2]
        y_map: [[[0.1,0.3,0.6], [0.5,0.3,0.2]],[[0.2,0.7,0.1], [0.2,0.3,0.5]]]
               -> [[0.1,0.3,0.6],[0.5,0.3,0.2],[0.2,0.7,0.1],[0.2,0.3,0.5]]
    '''
    y_true = y_true.reshape(-1, )
    h, w, c = y_map.shape
    y_map = y_map.reshape(h*w, c)
    mauc = metrics.roc_auc_score(y_true, y_map, multi_class='ovo')
    return mauc


def get_f1_iou_mcc_auc_fpr(y_true, y_pred, y_map):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)

    f1 = 2 * tp / (2 * tp + fp + fn)
    if np.isnan(f1):
        f1 = 0.

    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 0.00001)
    if np.isnan(mcc):
        mcc = 0.

    iou = tp / (fp + tp + fn)
    if np.isnan(iou):
        iou = 0.

    fpr = fp / (fp + tn)  # 假阳性率/虚警率
    if np.isnan(fpr):
        fpr = 0.

    auc = 0
    # 若一部分数据的标签全为0，即标签的类别只有一种，没有0和1两种，那么计算auc会报错
    try:
        auc = metrics.roc_auc_score(y_true.reshape(-1, ), y_map.reshape(-1, ))
        if np.isnan(auc):
            auc = 0.
    except ValueError:
        pass

    return f1, iou, mcc, auc, fpr
import csv
from numpy import *

cls_csv_all = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type1/others/docimg.csv"
cls_csv_sigal = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type1/test_result_917/5types_docimg.csv"

name = []
f11_dict = {}
iou1_dict = {}
mcc1_dict = {}
auc1_dict = {}
f12_dict = {}
iou2_dict = {}
mcc2_dict = {}
auc2_dict = {}
with open(cls_csv_all, encoding='utf-8-sig') as f:
    for row in csv.reader(f, skipinitialspace=True):
        row_name = row[0]
        name.append(row_name)
        f11_dict[row_name] = row[2]
        iou1_dict[row_name] = row[3]
        mcc1_dict[row_name] = row[4]
        auc1_dict[row_name] = row[5]
        f12_dict[row_name] = row[7]
        iou2_dict[row_name] = row[8]
        mcc2_dict[row_name] = row[9]
        auc2_dict[row_name] = row[10]
f = open(cls_csv_sigal, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['Type', 'f1-1', 'iou-1', 'mcc-1', 'auc-1', 'f1-2', 'iou-2', 'mcc-2', 'auc-2'])
f11 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
iou1 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
mcc1 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
auc1 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
f12 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
iou2 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
mcc2 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}
auc2 = {'com': [], 'spl': [], 'add': [], 'rem': [], 'rma': []}

for image_name in name:
    print(image_name)
    if 'com' in image_name:
        f11['com'].append(float(f11_dict[image_name]))
        iou1['com'].append(float(iou1_dict[image_name]))
        mcc1['com'].append(float(mcc1_dict[image_name]))
        auc1['com'].append(float(auc1_dict[image_name]))
        f12['com'].append(float(f12_dict[image_name]))
        iou2['com'].append(float(iou2_dict[image_name]))
        mcc2['com'].append(float(mcc2_dict[image_name]))
        auc2['com'].append(float(auc2_dict[image_name]))
    elif 'spl' in image_name:
        f11['spl'].append(float(f11_dict[image_name]))
        iou1['spl'].append(float(iou1_dict[image_name]))
        mcc1['spl'].append(float(mcc1_dict[image_name]))
        auc1['spl'].append(float(auc1_dict[image_name]))
        f12['spl'].append(float(f12_dict[image_name]))
        iou2['spl'].append(float(iou2_dict[image_name]))
        mcc2['spl'].append(float(mcc2_dict[image_name]))
        auc2['spl'].append(float(auc2_dict[image_name]))
    elif 'add' in image_name:
        f11['add'].append(float(f11_dict[image_name]))
        iou1['add'].append(float(iou1_dict[image_name]))
        mcc1['add'].append(float(mcc1_dict[image_name]))
        auc1['add'].append(float(auc1_dict[image_name]))
        f12['add'].append(float(f12_dict[image_name]))
        iou2['add'].append(float(iou2_dict[image_name]))
        mcc2['add'].append(float(mcc2_dict[image_name]))
        auc2['add'].append(float(auc2_dict[image_name]))
    elif 'rem' in image_name:
        f11['rem'].append(float(f11_dict[image_name]))
        iou1['rem'].append(float(iou1_dict[image_name]))
        mcc1['rem'].append(float(mcc1_dict[image_name]))
        auc1['rem'].append(float(auc1_dict[image_name]))
        f12['rem'].append(float(f12_dict[image_name]))
        iou2['rem'].append(float(iou2_dict[image_name]))
        mcc2['rem'].append(float(mcc2_dict[image_name]))
        auc2['rem'].append(float(auc2_dict[image_name]))
    elif 'rma' in image_name:
        f11['rma'].append(float(f11_dict[image_name]))
        iou1['rma'].append(float(iou1_dict[image_name]))
        mcc1['rma'].append(float(mcc1_dict[image_name]))
        auc1['rma'].append(float(auc1_dict[image_name]))
        f12['rma'].append(float(f12_dict[image_name]))
        iou2['rma'].append(float(iou2_dict[image_name]))
        mcc2['rma'].append(float(mcc2_dict[image_name]))
        auc2['rma'].append(float(auc2_dict[image_name]))
    else:
        pass

csv_writer.writerow(['com', mean(f11['com']), mean(iou1['com']), mean(mcc1['com']), mean(auc1['com']),
                     mean(f12['com']), mean(iou2['com']), mean(mcc2['com']), mean(auc2['com'])])
csv_writer.writerow(['spl', mean(f11['spl']), mean(iou1['spl']), mean(mcc1['spl']), mean(auc1['spl']),
                     mean(f12['spl']), mean(iou2['spl']), mean(mcc2['spl']), mean(auc2['spl'])])
csv_writer.writerow(['add', mean(f11['add']), mean(iou1['add']), mean(mcc1['add']), mean(auc1['add']),
                     mean(f12['add']), mean(iou2['add']), mean(mcc2['add']), mean(auc2['add'])])
csv_writer.writerow(['rem', mean(f11['rem']), mean(iou1['rem']), mean(mcc1['rem']), mean(auc1['rem']),
                     mean(f12['rem']), mean(iou2['rem']), mean(mcc2['rem']), mean(auc2['rem'])])
csv_writer.writerow(['rma', mean(f11['rma']), mean(iou1['rma']), mean(mcc1['rma']), mean(auc1['rma']),
                     mean(f12['rma']), mean(iou2['rma']), mean(mcc2['rma']), mean(auc2['rma'])])

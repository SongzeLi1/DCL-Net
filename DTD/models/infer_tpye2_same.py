import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
warnings.filterwarnings('ignore')
from utils.utils import inial_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from dtd import *
from utils.segmetric import *
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageFile
from utils.metric import *
import time
import logging
from dataset.DOCDataset_gt2s import DOCDataset

Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def infer_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size is not None:
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),  # width, height
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])



device = 'cuda'
n_class = 2
mask_threshold, viz, save_map, save_mask, cal_metrics, batch_size = 0.5, True, True, True, False, 160
input_size = [512, 512]
patch_size, patch_stride = None, None
tta_scale = None
# ---------------
image_dataset_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_images/'
gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_gt3/'
test_img_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/test_images_75_100/'  # '/pubdata/1/docimg/docimg_split811/crop512x512/test_images_crop512stride256/'
test_gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/test_gt3_crop512stride256/'
checkpoint_dir = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/dtd_512x512/fold_0_best-ckpt.pth"
save_results_path = '/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/test_result_917/'
# ---------------
map_path = save_results_path + 'map{}x{}/'.format(input_size[0], input_size[1])
pred_path = save_results_path + 'pred{}x{}/'.format(input_size[0], input_size[1])
if (os.path.exists(map_path) == False): os.makedirs(map_path)
if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
logger = inial_logger(os.path.join(save_results_path, 'test.log'))
filenames = os.listdir(test_img_path)
filenames.sort()
# filenames = filenames
mious, f1forgerys, iouforgerys = [], [], []
t1 = time.time()

# ------infer_batch start------
logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_mask:{}, cal_metrics:{}, batch_size:{}'.format(n_class,
                                                                                                       mask_threshold,
                                                                                                       viz,
                                                                                                       save_mask,
                                                                                                       cal_metrics,
                                                                                                       batch_size))
logger.info(
    'test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames), input_size,
                                                                                      patch_size, patch_stride,
                                                                                      tta_scale))
logger.info('checkpoint:{}'.format(checkpoint_dir))
logger.info('test_img_path:{}'.format(test_img_path))
logger.info('test_gt_path:{}'.format(test_gt_path))
logger.info('save_test_results_path:{}'.format(save_results_path))
logger.info(
    '======================================================================================================')
model = seg_dtd('', 2).cuda()
model = torch.nn.DataParallel(model)
checkpoints = torch.load(checkpoint_dir)
if 'state_dict' in checkpoints.keys():
    model.load_state_dict(checkpoints['state_dict'])
else:
    model.load_state_dict(checkpoints)
model.eval()
test_data = DOCDataset(filenames, test_img_path, test_gt_path, infer_transform(input_size))
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
print('Total Test Batches: {}'.format(len(test_loader)))

# save as LA
for batch_idx, batch_samples in enumerate(test_loader):
    img, dct, qs = batch_samples['image'], batch_samples['dct'], batch_samples['qs']
    img, dct, qs = Variable(img.to(device)), Variable(dct.to(device)), Variable(qs.unsqueeze(1).to(device))
    print('Testing batch {} ......'.format(batch_idx))
    with torch.no_grad():
        out = model(img, dct, qs)  # DTD
    out = torch.sigmoid(out).squeeze().cpu().numpy()
    pred = np.argmax(out, axis=1)
    # out = F.softmax(out, dim=1)
    # out = out.cpu().data.numpy()
    for i in range(pred.shape[0]):
        img_name = filenames[batch_idx * batch_size + i]
        if save_map:
            map = np.array(out[i]).transpose(1, 2, 0)
            map = Image.fromarray(np.uint8(map * 255))
            map = map.convert('LA')
            map.save(map_path + img_name[:-4] + '.png')
        if save_mask:
            pred3 = pred[i]
            pred3[pred3 == 1] = 255
            pred3 = Image.fromarray(np.uint8(pred3))
            pred3 = pred3.convert('L')
            pred3.save(pred_path + img_name[:-4] + '.png')
print('test time:', time.time()-t1)

# ----------------merge----------------------
from utils.merge import get_row, get_col

dst_dir = save_results_path
block_read_path = dst_dir + 'pred512x512/'
merge_write_path = dst_dir + 'predblockmerge/'
mapblock_read_path = dst_dir + 'map512x512/'
mapmerge_write_path = dst_dir + 'mapblockmerge/'

if not os.path.exists(merge_write_path): os.makedirs(merge_write_path)
if not os.path.exists(mapmerge_write_path): os.makedirs(mapmerge_write_path)
image_name_list = os.listdir(image_dataset_path)    # test image
block_name_list = os.listdir(block_read_path)
mapblock_name_list = os.listdir(mapblock_read_path) # block name

block_size = 512
step = 256
image_name_list.sort()
print('merge num:{}'.format(len(image_name_list)))
# print(image_name_list)


for image_name in image_name_list:
    print(image_name)
    image = cv2.imread(image_dataset_path + image_name, cv2.IMREAD_UNCHANGED)
    height, width = image.shape[0], image.shape[1]
    channel = 2
    block_name = [block for block in block_name_list if image_name.split(".", -1)[0] in block]
    mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] in mapblock]
    block_name.sort()
    mapblock_name.sort()

    # ---合并概率图---
    mapmerge_block = np.zeros((height, width, channel))
    maptimes_save = np.zeros((height, width, channel))
    for mapblock in mapblock_name:
        mappredict_block = Image.open(mapblock_read_path + mapblock).convert('LA')
        mappredict_block = np.array(mappredict_block, dtype=np.uint8)
        # mappredict_block = np.load(mapblock_read_path + mapblock).transpose(1, 2, 0)
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
    # np.save(mapmerge_write_path + image_name[:-4] + '.npy', mapmerge_block)
    mapmerge_block_ = Image.fromarray(np.uint8(mapmerge_block))
    mapmerge_block_ = mapmerge_block_.convert('LA')
    mapmerge_block_.save(mapmerge_write_path + image_name[:-4] + '.png')

    # ---合并二值图---
    pred = np.argmax(mapmerge_block, axis=2)
    pred[pred == 1] = 255
    pred3 = Image.fromarray(np.uint8(pred))
    pred3 = pred3.convert('L')
    pred3.save(merge_write_path + image_name[:-4] + '.png')
print("finish merge")


# --------------cal metrics------------------------
import csv

save_metrics_path = save_results_path
metrics_csv = save_metrics_path + "2kinds_metric_same_type2.csv"
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['img','gt','f1','iou','mcc','auc','precision','tpr_recall','tnr','fpr','fnr'])
f.close()
filenames = os.listdir(merge_write_path)
filenames.sort()
f1s, ious, mccs, aucs, precisions, tprs, tnrs, fprs, fnrs= [], [], [], [], [], [], [], [], []
for i in range(len(filenames)):
    name = filenames[i]
    map = Image.open(mapmerge_write_path + name).convert('LA')
    map = np.array(map) / 255
    # map = np.load(mapmerge_write_path + name.replace('.png', '.npy'))
    # map = map / 255
    map_forgery = map[:, :, 1]
    # map_mosaic = map[:, :, 2]

    pred = Image.open(merge_write_path + name).convert('L')
    pred = np.array(pred)
    pred[pred == 255] = 1
    if gt_path is not None:
        gt_name = name.replace('psc_', 'gt3_')
        gt = Image.open(gt_path + gt_name).convert('L')
        gt = np.array(gt)
        gt[gt == 255] = 0
        gt[gt == 76] = 1
        gt[gt == 29] = 0
    else:
        gt_name = ""
        h, w = pred.shape
        gt = np.zeros((h, w), dtype=np.uint8)

    tpr_recall, tnr, fpr, fnr, precision, f1, mcc, iou, tn, tp, fn, fp, auc = get_metrics(gt, pred, map_forgery)
    logger.info('{}, {}, f1:{:.3f}, iou:{:.3f}, mcc:{:.3f}, fpr:{:.3f}, auc:{:.3f}'.format(i, name, f1, iou, mcc, fpr, auc))
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
logger.info('test num:{}, f1:{:.3f}, iou:{:.3f}, mcc:{:.3f}, fpr:{:.3f}, auc:{:.3f}'.format(len(filenames), f1_mean, iou_mean, mcc_mean, fpr_mean, auc_mean))
f = open(metrics_csv, 'a+', newline='')
csv_writer = csv.writer(f)
csv_writer.writerow(['Average', len(filenames), f1_mean, iou_mean, mcc_mean, auc_mean, precision_mean, tpr_mean, tnr_mean, fpr_mean, fnr_mean])
csv_writer.writerow(['END','END','f1','iou','mcc','auc','precision','tpr_recall','tnr','fpr','fnr'])
f.close()

print('all time:', time.time() - t1)

import csv
import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
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

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    n_class = 2
    mask_threshold, viz, save_map, save_mask, batch_size = None, False, True, True, 160
    input_size = [512, 512] # None
    patch_size, patch_stride = None, None
    tta_scale = None  # [128, 192, 256]
    test_img_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/test_images_75_100/'
    test_gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/test_gt3_crop512stride256/'
    checkpoint_dir = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/dtd_512x512/fold_0_best-ckpt.pth"
    save_results_path = '/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/test_result_type2/'
    map_path = save_results_path + 'map{}x{}/'.format(input_size[0], input_size[1])
    pred_path = save_results_path + 'pred{}x{}/'.format(input_size[0], input_size[1])
    if (os.path.exists(map_path) == False): os.makedirs(map_path)
    if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
    logger = inial_logger(os.path.join(save_results_path, 'test_{}x{}.log'.format(input_size[0], input_size[1])))
    filenames = os.listdir(test_img_path)
    filenames.sort()
    # filenames = filenames[0*3232:1*3232]
    # mious, f1forgerys, iouforgerys, f1mosaics, ioumosaics = [], [], [], [], []
    f1s, mccs, ious, = [], [], []
    t1 = time.time()
    # ------infer_batch start------
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_mask:{}, batch_size:{}'.format(n_class,mask_threshold,viz,save_mask,batch_size))
    logger.info(
        'test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames),input_size,patch_size,patch_stride,tta_scale))
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
    device = 'cuda'
    print('Total Test Batches: {}'.format(len(test_loader)))
    metrics_csv = save_results_path + "512x512_forgerymosaic_samekind_metric.csv"
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['f1', 'iou', 'mcc'])
    f.close()
    f1forgerys, iouforgerys = [], []
    for batch_idx, batch_samples in enumerate(test_loader):     # important!!! gt should from dataloader:Datasets!!
        img, gt, dct, qs = batch_samples['image'], batch_samples['label'], batch_samples['dct'], batch_samples['qs']
        img, gt, dct, qs = Variable(img.to(device)), Variable(gt.to(device)), Variable(dct.to(device)), Variable(qs.unsqueeze(1).to(device))
        print('Testing batch {} ......'.format(batch_idx))
        gt = gt.to(torch.int64)

        with torch.no_grad():
            predicts = model(img, dct, qs)
        predicts = predicts.argmax(1)
        predicts = predicts.cpu().data.numpy()
        gt = gt.cpu().data.numpy()
        tpr_recall, tnr, fpr, fnr, precision, f1, mcc, iou, tn, tp, fn, fp = \
            get_metrics_without_auc(gt, predicts)

        f = open(metrics_csv, 'a+', newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow([f1, iou, mcc])
        f.close()

        f1s.append(f1)
        mccs.append(mcc)
        ious.append(iou)

    f1_mean, iou_mean, mcc_mean = \
        np.mean(f1s), np.mean(ious), np.mean(mccs)

    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['Average', len(filenames), f1_mean, iou_mean, mcc_mean])
    csv_writer.writerow(['END', 'END', 'f1', 'iou', 'mcc'])
    f.close()
    print('test time:', time.time() - t1)


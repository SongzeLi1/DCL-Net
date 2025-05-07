import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import warnings
from models.mvssnet import get_mvss
import numpy as np
import torch
import cv2
from PIL import Image, ImageFile
import time
import logging
import csv
from metric import *
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

def infer_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size is not None:
        return A.Compose([
            A.Resize(input_size[0], input_size[1]), # width, height
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])

class DOCDataset(Dataset):
    def __init__(self, names, imgs_dir, masks_dir, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.names = names
        logging.info(f'Creating dataset with {len(self.names)} examples')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        image = Image.open(self.imgs_dir + name).convert('RGB')
        if self.masks_dir is not None:
            gtname = name.replace('psc_', 'gt3_')
            gtname = gtname.replace('.jpg', '.png')
            gtname = gtname.replace('.tif', '.png')
            gtname = gtname.replace('_qf60', '')
            gtname = gtname.replace('_qf70', '')
            gtname = gtname.replace('_qf80', '')
            gtname = gtname.replace('_qf90', '')
            mask = Image.open(self.masks_dir + gtname).convert('L')
        else:
            mask = Image.open(self.imgs_dir + name).convert('RGB').convert('L')
        # image = cv2.imread(img_file[0], cv2.IMREAD_COLOR) # 不含alpha通道
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.imread(img_file[0], cv2.IMREAD_UNCHANGED) # 含alpha通道
        # mask = cv2.imread(mask_file[0], cv2.IMREAD_GRAYSCALE)

        image = np.array(image, np.uint8)
        mask = np.array(mask)
        # print(mask.min(), mask.max())
        # print(image.shape, ocrlabel.shape, mask.shape)

        # 二分类
        # if mask.max() > 1: mask = np.uint8(mask / 255)
        # # ---三分类 docimg---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 2
        # # ---二分类 docimg Tamper和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 1
        # # ---二分类 docimg Orig和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 0
        # ---二分类 Alinew, SUPATLANTIQUE---
        mask[mask !=0] = 1

        # print(np.array(image).min(), np.array(image).max()) # 0, 255
        # print(np.array(mask).min(), np.array(mask).max()) # 0, 1
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': mask
        }


if __name__=="__main__":
    warnings.filterwarnings('ignore')
    n_class = 1
    mask_threshold, viz, save_pmap, save_mask, batch_size = 0.5, True, True, True, 1 # batch_size只能为1
    input_size = [768, 768]  # None [1792, 1792] [1024, 1024], [1440, 1440], [768, 768], [512, 512]
    patch_size, patch_stride = None, None
    tta_scale = None  # [128, 192, 256]
    # test_img_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
    # test_gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/' # None '/pubdata/zhengkengtao/docimg/docimg_split811/test_gt3/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/test_gt/'
    test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/test_imgs/'
    test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/test_gt/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/img/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_gt/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split415/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split415/test_gt/'
    # test_img_path = '/data1/zhengkengtao/docimg/tamper_mosaic/'
    # test_gt_path = '/data1/zhengkengtao/docimg/gt3/'
    # test_img_path = '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride256/'
    # test_gt_path = '/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_gt3_crop512stride256/'
    # checkpoint_dir = '/data1/zhengkengtao/exps/0726_MVSSNet_Alinewtrain_split19_pretrainwithdocimgsplit811_crop512x512/26_acc_0.9_f1_0.259_lr_0.0001.pkl'
    # save_results_path = '/data1/zhengkengtao/exps/0726_MVSSNet_Alinewtrain_split19_pretrainwithdocimgsplit811_crop512x512/pred_{}x{}/'.format(input_size[0], input_size[1])
    checkpoint_dir = '/data1/zhengkengtao/exps/1617_MVSS_Alinew_trainsplit118_noAug_pretrainwithdocimgsplit811Aug/7_acc_0.901_f1_0.216_lr_0.0001.pkl'
    save_results_path = '/data1/zhengkengtao/exps/1617_MVSS_Alinew_trainsplit118_noAug_pretrainwithdocimgsplit811Aug/testepoch7_Alinew3200/'
    pmap_path = save_results_path + 'unthreshold/'
    pred_path = save_results_path + 'threshold_{}/'.format(mask_threshold)
    if (os.path.exists(pmap_path) == False): os.makedirs(pmap_path)
    if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
    logger = inial_logger(os.path.join(save_results_path, 'test.log'))
    filenames = os.listdir(test_img_path)
    filenames.sort()
    # filenames = filenames[:3]
    mious, f1forgerys, iouforgerys, f1mosaics, ioumosaics = [], [], [], [], []
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_pmap:{}, save_mask:{}, batch_size:{}'.format(n_class,mask_threshold, viz, save_pmap, save_mask, batch_size))
    logger.info('test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames), input_size, patch_size, patch_stride, tta_scale))
    logger.info('checkpoint:{}'.format(checkpoint_dir))
    logger.info('test_img_path:{}'.format(test_img_path))
    logger.info('test_gt_path:{}'.format(test_gt_path))
    logger.info('save_test_results_path:{}'.format(save_results_path))
    logger.info('======================================================================================================')
    model = get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3)
    model.cuda()
    model = torch.nn.DataParallel(model)
    checkpoints = torch.load(checkpoint_dir)
    # for k, v in checkpoints.items():
    #     print('checkpoint:', k)
    # model_ = model.state_dict()
    # for k, v in model_.items():
    #     print('model:', k)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    model.eval()
    test_data = DOCDataset(filenames, test_img_path, test_gt_path, infer_transform(input_size))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    out_lst = []
    t1 = time.time()
    print('Total Test Batches: {}'.format(len(test_loader)))
    for batch_idx, batch_samples in enumerate(test_loader):
        img = batch_samples['image'].cuda()
        print('Testing batch {} ......'.format(batch_idx))
        with torch.no_grad():
            _, out = model(img)
        # 二分类
        # preds = (torch.sigmoid(out) > mask_threshold).squeeze().int().detach().cpu().numpy()
        out = torch.sigmoid(out).squeeze().cpu().numpy()
        # print(out.shape) # [b,h,w]
        # # # 三分类
        # out = F.softmax(out, dim=1)  # [b,c,h,w]
        # out = out.cpu().data.numpy() # [b,c,h,w]
        # preds = np.argmax(out, axis=1)  # [b,h,w]
        # print(preds.shape,preds.min(),preds.max()) # [b,h,w], 0-2
        out_lst.append(out)
    # np.save(save_results_path + 'pmap_results.npy', np.array(out_lst)) # 将测试结果保存为npy
    # # out_lst = np.load(save_results_path + 'pmap_results.npy')
    for i in range(len(filenames)):
        out = out_lst[i]
        img_name = filenames[i]
        print(img_name)
        image = cv2.imread(test_img_path + img_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ht, wd, _ = image.shape
        if input_size is not None:
            out = cv2.resize(out, (wd, ht), interpolation=cv2.INTER_NEAREST)
        # print(pred.shape, pred.min(), pred.max()) # [h,w] 0, 2
        # pred = np.int64(pred)
        if save_pmap:
            pmap = Image.fromarray(np.uint8(out * 255))
            pmap = pmap.convert('L')
            pmap.save(pmap_path + img_name[:-4] + '.png')
        if save_mask:
            pred = np.array(out>mask_threshold, dtype=np.uint8)
            pred = Image.fromarray(np.uint8(pred * 255))
            pred = pred.convert('L')
            pred.save(pred_path + img_name[:-4] + '.png')
    print('test time:', time.time()-t1)


# # ---------------------------------------------------------------------------------
# 输出一通道——篡改脱敏视为一类计算指标
# gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
# gt_path = '/data1/zhengkengtao/docimg/gt3/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
# gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
# testresult_dir = '/data1/zhengkengtao/exps/1509_MVSSNet_Alinew_split415_ft_with_docimgsplit811/test_Alinewsplit415test_768x768/'
gt_path = test_gt_path
testresult_dir = save_results_path
map_path = pmap_path
pred_path = pred_path
# map_path = testresult_dir + 'mapblockmerge/'
# pred_path = testresult_dir +  'predblockmerge_thres0.5/'
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



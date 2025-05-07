import os
import warnings

import jpegio

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '6, 7'
warnings.filterwarnings('ignore')
from utils.utils import inial_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from models.dtd_cls import *
import glob
import torch
from PIL import Image, ImageFile
from utils.metric import *
import time
import csv
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_tn_tp_fn_fp(y_true, y_pred):
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred))).astype(np.float64)
    tp = np.sum(np.logical_and(               y_true ,                y_pred )).astype(np.float64)
    fn = np.sum(np.logical_and(               y_true , np.logical_not(y_pred))).astype(np.float64)
    fp = np.sum(np.logical_and(np.logical_not(y_true),                y_pred )).astype(np.float64)
    return tn, tp, fn, fp

def get_metrics(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    if np.isnan(acc):
        acc = 0.
    precision = tp / (tp + fp)
    if np.isnan(precision):
        precision = 0.
    recall = tp / (tp + fn)
    if np.isnan(recall):
        recall = 0.
    f1 = 2 * tp / (2 * tp + fp + fn)
    if np.isnan(f1):
        f1 = 0.
    iou = tp / (fp + tp + fn)
    if np.isnan(iou):
        iou = 0.
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 0.00001)
    if np.isnan(mcc):
        mcc = 0.
    fpr = fp / (fp + tn)  # 假阳性率/虚警率
    if np.isnan(fpr):
        fpr = 0.
    return acc, precision, recall, f1, iou, mcc, fpr


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
        self.transform = transform
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.names = names
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        image = Image.open(self.imgs_dir + name).convert('RGB')
        image = np.array(image, np.uint8)

        jpg_dct = jpegio.read(self.imgs_dir + name)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()
        dct = torch.LongTensor(dct_ori)
        qs = torch.LongTensor(use_qtb2)

        if 'psc_' in name or 'ps_' in name:
            gt3name = name.replace('psc_', 'gt3_psc_')
            gt3name = gt3name.replace('ps_', 'gt3_ps_')
            gt3name = gt3name.replace('.jpg', '.png')
            gt3 = Image.open(self.masks_dir + gt3name).convert('L')
            gt3 = np.array(gt3, np.uint8)
            gt3[gt3 == 255] = 0
            gt3[gt3 == 76] = 1
            gt3[gt3 == 29] = 2
            if 1 in gt3:
                clsgt = 1
            else:
                clsgt = 0
        else:
            clsgt = 0
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return {
            'image': image,
            'clsgt': clsgt,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qs': qs
        }


if __name__=="__main__":
    n_class = 1
    device = 'cuda'
    mask_threshold, viz, save_map, save_mask, batch_size = None, False, False, False, 2*30
    input_size = [512, 512]
    stride = 512
    checkpoint_dir = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/cls_2class_102/dtd_512x512/fold_0_best-ckpt.pth"
    save_results_path = '/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/Findit_directly/cls_2class/test_result_106/'
    if (os.path.exists(save_results_path) == False): os.makedirs(save_results_path)
    cls_csv = save_results_path + 'crop512stride{}_cls.csv'.format(stride)
    cls_csv_all = save_results_path + 'crop512stride{}_cls_all.csv'.format(stride)
    cls_csv_all_cal = save_results_path + 'crop512stride{}_cls_all_cal.csv'.format(stride)
    logger = inial_logger(os.path.join(save_results_path, 'test_{}x{}.log'.format(input_size[0], input_size[1])))
    # 图像块
    test_img_path = "/pubdata/lisongze/docimg/exam/docimg2jpeg/findit/imgs_crop512stride512/"  # '/pubdata/lisongze/docimg/exam/docimg2jpeg/test_images_75_100/'
    test_gt_path = "/pubdata/lisongze/docimg/exam/docimg2jpeg/findit/gt3_crop512stride512/"
    filenames = "/pubdata/lisongze/docimg/exam/docimg2jpeg/findit/imgs_crop512stride512/"
    filenames = os.listdir(filenames)
    filenames.sort()
    # 整图
    image_name_list = "/pubdata/lisongze/docimg/exam/docimg2jpeg/findit/imgs/"
    image_name_list = os.listdir(image_name_list)
    image_name_list.sort()
    t1 = time.time()

    '''
    # ------infer_batch start------
    f = open(cls_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['name', 'gt', 'clspred'])
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_mask:{}, batch_size:{}'.format(n_class,mask_threshold, viz, save_mask, batch_size))
    logger.info('test_num:{}, input_size:{}'.format(len(filenames), input_size))
    logger.info('checkpoint:{}'.format(checkpoint_dir))
    logger.info('save_test_results_path:{}'.format(save_results_path))
    logger.info('======================================================================================================')
    model = seg_dtd()
    model.cuda()
    model = torch.nn.DataParallel(model)
    checkpoints = torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoints, strict=False)
    model.eval()
    test_data = DOCDataset(filenames, test_img_path, test_gt_path, infer_transform(input_size))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    print('Total Test Batches: {}'.format(len(test_loader)))
    print('Full img num:', len(image_name_list))
    accs = 0
    for batch_idx, batch_samples in enumerate(test_loader):
        data, dct, qs, clsgt = batch_samples['image'], batch_samples['dct'], batch_samples['qs'], batch_samples['clsgt']
        data, dct_coef, qs, clsgt = Variable(data.to(device)), Variable(dct.to(device)), Variable(qs.unsqueeze(1).to(device)), Variable(clsgt.to(device))

        clsgt = clsgt.to(torch.int64)
        print('Testing img {} ......'.format(batch_idx))
        with torch.no_grad():
            clspred = model(data, dct_coef, qs)
        clspred = torch.sigmoid(clspred)
        clspred = clspred > 0.5
        for i in tqdm(range(clsgt.shape[0])):
            acc = torch.eq(clspred[i], clsgt[i]).sum().float().item()
            accs += acc
            csv_writer.writerow([filenames[i + batch_size*batch_idx], clsgt[i].item(), int(clspred[i])])

    acc = accs / len(filenames)
    print('ACC: {:.5f}'.format(acc))
    csv_writer.writerow(['Patch Average Acc', len(filenames), acc])
    f.close()
    

    # # 统计表格得到ACC
    dict = {}
    with open(cls_csv, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            row0 = row[0]
            dict[row0] = row[2]
    f = open(cls_csv_all, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['NAME', 'GT', 'PRED'])
    accuracy = 0
    imggt_list, imgpred_list = [], []
    for image_name in image_name_list:
        if 'orig' in image_name or 'mosaic' in image_name:
            imggt = 0
        else:
            imggt = 1
        image_name_ = image_name.split('/')[-1]
        print(image_name_)
        filenames_ = [blockname.split('/')[-1] for blockname in filenames]
        preds = [dict[blockname] for blockname in filenames_ if image_name_.split(".", -1)[0] == blockname[0:blockname[0:blockname.rfind('_')].rfind('_')]]
        print(preds)
        if '1' in preds:
            imgpred = 1
        else:
            imgpred = 0
        csv_writer.writerow([image_name, imggt, imgpred])

        imggt_list.append(imggt)
        imgpred_list.append(imgpred)

    imggt_list = np.array(imggt_list)
    imgpred_list = np.array(imgpred_list)
    csv_writer.writerow(['NUMS', 'acc', 'precision', 'recall', 'f1', 'iou', 'mcc', 'fpr'])
    acc, precision, recall, f1, iou, mcc, fpr = get_metrics(imggt_list, imgpred_list)
    csv_writer.writerow([len(image_name_list), acc, precision, recall, f1, iou, mcc, fpr])
    print('Average Acc: {:.5f}'.format(acc))
    print('time:', time.time() - t1)
    '''

    # # 统计表格得到ACC
    pred_dict = {}
    gt_dict = {}
    with open(cls_csv_all, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            row0 = row[0]
            print(row0)
            gt_dict[row0] = row[1]
            pred_dict[row0] = row[2]
    f = open(cls_csv_all_cal, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Type', 'TamperMosaic', 'Tamper', 'Orig', 'Mosaic'])
    TamperMosaic = 0
    Tamper = 0
    Orig = 0
    Mosaic = 0
    for image_name in image_name_list:
        image_name = image_name.split('/')[-1]
        print(image_name)
        if 'mosaic' in image_name:
            if gt_dict[image_name] == pred_dict[image_name]:
                Mosaic += 1
        elif 'psc' in image_name:
            if gt_dict[image_name] == pred_dict[image_name]:
                TamperMosaic += 1
        elif 'ps' in image_name:
            if gt_dict[image_name] == pred_dict[image_name]:
                Tamper += 1
        else:
            if gt_dict[image_name] == pred_dict[image_name]:
                Orig += 1

    csv_writer.writerow(['Pred_True', TamperMosaic, Tamper, Orig, Mosaic])


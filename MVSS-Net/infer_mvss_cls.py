import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.utils import inial_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import warnings
from networks.mvss_net.mvssnet_ import *
from networks.mvss_net.mvssnet_cls import *
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
    def __init__(self, names, transform):
        self.transform = transform
        self.names = names
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        image = Image.open(name).convert('RGB')
        image = np.array(image, np.uint8)
        if 'psc_' in name or 'ps_' in name:
            gt3name = name.replace('_images', '_gt3')
            gt3name = gt3name.replace('psc_', 'gt3_')
            gt3name = gt3name.replace('_tamper', '_gt3')
            gt3name = gt3name.replace('ps_', 'gt3_')
            gt3name = gt3name.replace('.jpg', '.png')
            gt3name = gt3name.replace('.JPG', '.png')
            gt3 = Image.open(gt3name).convert('L')
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
            'clsgt': clsgt
        }


def get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, n_input=3, constrain=True, **kwargs):
    model = MVSSNet(nclass, backbone=backbone,
                    pretrained_base=pretrained_base,
                    sobel=sobel,
                    n_input=n_input,
                    constrain=constrain,
                    **kwargs)
    return model

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    n_class = 1
    mask_threshold, viz, save_pmap, save_mask, batch_size = 0.5, True, False, False, 1 # batch_size只能为1
    input_size = [512, 512]  # None [1792, 1792] [1024, 1024], [1440, 1440], [768, 768], [512, 512]
    patch_size, patch_stride = None, None
    tta_scale = None  # [128, 192, 256]
    stride = 512
    # checkpoint_dir = '/data1/zhengkengtao/exps/0723_MVSSNet_docimg_split811_png/63_acc_0.96_f1_0.707_lr_0.0001.pkl'
    # save_results_path = '/data1/zhengkengtao/exps/0723_MVSSNet_docimg_split811_png/' \
    #                     'testepoch63_docimgsplit811_Orig200Psc200Mosaic200Ps200_crop512stride{}/'.format(stride)
    # checkpoint_dir = '/data1/zhengkengtao/exps/1427_mvss_docimgsplit811_crop512train_Cls_noAug/7_acc_1.951_f1_0.56_clsacc_0.865_lr_0.0001.pkl'
    # save_results_path = '/data1/zhengkengtao/exps/1427_mvss_docimgsplit811_crop512train_Cls_noAug/' \
    #                     'testepoch7_docimgsplit811_Orig200Psc200Mosaic200Ps200_crop512stride{}/'.format(stride)
    checkpoint_dir = '/data1/zhengkengtao/exps/1427_mvss_docimgsplit811_crop512train_Cls_noAug_adddata/9_acc_1.971_clsacc_0.876_lr_0.0001.pkl'
    save_results_path = '/data1/zhengkengtao/exps/1427_mvss_docimgsplit811_crop512train_Cls_noAug_adddata/' \
                        'testepoch9_docimgsplit811_Orig200Psc200Mosaic200Ps200_crop512stride{}/'.format(stride)
    if (os.path.exists(save_results_path) == False): os.makedirs(save_results_path)
    cls_csv = save_results_path + 'crop512stride{}_cls.csv'.format(stride)
    logger = inial_logger(os.path.join(save_results_path, 'test_{}x{}.log'.format(input_size[0], input_size[1])))
    # 图像块
    # filenames = glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_crop512stride128/*.png') + \
    #             glob.glob('/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride128/*.png')

    filenames = glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_crop512stride{}/*.png'.format(stride)) + \
                glob.glob('/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride{}/*.png'.format(stride)) + \
                glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic_crop512stride{}/*.png'.format(stride)) + \
                glob.glob('/data1/zhengkengtao/docimg/docimg_split811/test_tamper_crop512stride{}/*.png'.format(stride))
    filenames.sort()
    # 整图
    image_name_list = glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig/*.jpg') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig/*.JPG') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/test_images/*.png') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic/*.png') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/test_tamper/*.png')
    image_name_list.sort()
    t1 = time.time()

    f = open(cls_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['name', 'gt', 'clspred'])
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_mask:{}, batch_size:{}'.format(n_class, mask_threshold, viz, save_mask, batch_size))
    logger.info('test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames), input_size, patch_size, patch_stride, tta_scale))
    logger.info('checkpoint:{}'.format(checkpoint_dir))
    logger.info('save_test_results_path:{}'.format(save_results_path))
    logger.info('======================================================================================================')

    model = get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3)
    model.cuda()
    model = torch.nn.DataParallel(model)
    checkpoints = torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    model.eval()
    test_data = DOCDataset(filenames, infer_transform(input_size))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    print('Total Test Batches: {}'.format(len(test_loader)))
    print('Full img num:', len(image_name_list))
    accs = 0
    for batch_idx, batch_samples in enumerate(test_loader):
        img = batch_samples['image'].cuda()
        clsgt = batch_samples['clsgt'].cuda()
        clsgt = clsgt.to(torch.int64)
        print('Testing batch {} ......'.format(batch_idx))
        with torch.no_grad():
            # _, out = model(img)
            _, out, cls = model(img)
        # # preds = (torch.sigmoid(out) > mask_threshold).squeeze().int().detach().cpu().numpy()
        # # print(out.min(), out.max())
        # out = torch.sigmoid(out)
        # # print(out.min(), out.max())
        # out = out.max()
        # # print(out)
        # clspred = out > 0.5
        # print(clspred)
        clspred = cls > 0.5
        acc = torch.eq(clspred, clsgt).sum().float().item()
        accs += acc
        csv_writer.writerow([filenames[batch_idx], clsgt.item(), int(clspred)])
    acc = accs / len(filenames)
    print('ACC: {:.5f}'.format(acc))
    csv_writer.writerow(['Patch Average Acc', len(filenames), acc])
    f.close()

    # # 统计表格得到ACC
    dict = {}
    with open(cls_csv, encoding='utf-8-sig') as f:
        for row in csv.reader(f, skipinitialspace=True):
            # print(row)
            row0 = row[0].split('/')[-1]
            dict[row0] = row[2]
    f = open(cls_csv, 'a+', newline='')
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
        preds = [int(dict[blockname]) for blockname in filenames_ if
                 image_name_.split(".", -1)[0] == blockname[0:blockname[0:blockname.rfind('_')].rfind('_')]]
        print(preds)
        if 1 in preds:
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
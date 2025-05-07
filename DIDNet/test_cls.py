import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from models import *
import argparse
import csv
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import warnings
from PIL import ImageFile
import glob
import time
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_tn_tp_fn_fp(y_true, y_pred):
    tn = np.sum(np.logical_and(np.logical_not(y_true), np.logical_not(y_pred))).astype(np.float64)
    tp = np.sum(np.logical_and(y_true, y_pred)).astype(np.float64)
    fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred))).astype(np.float64)
    fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred)).astype(np.float64)
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

def get_acc(y_true, y_pred):
    tn, tp, fn, fp = get_tn_tp_fn_fp(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    if np.isnan(acc):
        acc = 0.
    return acc

def SRM(imgs):
    # SQUARE 5×5
    filter1 = [[-1, 2, -2, 2, -1],
               [2, -6, 8, -6, 2],
               [-2, 8, -12, 8, -2],
               [2, -6, 8, -6, 2],
               [-1, 2, -2, 2, -1]]
    # SQUARE 3×3   
    filter2 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    ## Vertical second-order
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, -2, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    ## Horizontal second-order
    filter4 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 1, -2, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    ## Horizontal first-order
    filter5 = [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, -1, 1, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]
    ## Vertical first-order
    filter6 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, -1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]]

    filter1 = np.asarray(filter1, dtype=float) / 12
    filter2 = np.asarray(filter2, dtype=float) / 4
    filter3 = np.asarray(filter3, dtype=float) / 2
    filter4 = np.asarray(filter4, dtype=float) / 2
    filter5 = np.asarray(filter5, dtype=float)
    filter6 = np.asarray(filter6, dtype=float)

    filters = []
    filters = [[filter1, filter1, filter1], [filter2, filter2, filter2], [filter3, filter3, filter3],
    [filter4, filter4, filter4], [filter5, filter5, filter5], [filter6, filter6, filter6]]  # (3,3,5,5)
    filters = torch.FloatTensor(filters)    # (3,3,5,5)
    imgs = np.array(imgs, dtype=float)  # (375,500,3)
    imgs = np.einsum('klij->kjli', imgs)
    input = torch.tensor(imgs, dtype=torch.float32)

    op1 = F.conv2d(input, filters, stride=1, padding=2)
    op1 = op1[0]
    op1 = np.round(op1)
    op1[op1 > 2] = 2
    op1[op1 < -2] = -2
    return op1

class DataSetLoader(Dataset):
    def __init__(self, dataList):
        super(DataSetLoader, self).__init__()
        self.dataList = dataList
    def __getitem__(self, index):
        image = Image.open(self.dataList[index]).convert('RGB')
        image_name = self.dataList[index].split("/", -1)[-1]
        imageArray = np.asarray(image)
        srm = SRM([imageArray])
        srm = np.einsum('jkl->klj', srm).astype(np.uint8)


        image = ToTensor()(image)
        srm = ToTensor()(srm)
        return image, image_name, srm
    def __len__(self):
        return len(self.dataList)
def read_list(path):
    pathlist = []
    files = os.listdir(path)
    for file in files:
        pathlist.append(os.path.join(path, file))
    return pathlist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=1) # 只能设置为1，否则出错
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--step_size', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.9)
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86_map/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_epoch86/")
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/mapblock/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/test_Alinewtrainallepoch86/block/")
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/mapblock/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/0717_DIDNet_Alinew_train_split811/test_docimgall_epoch10/block/")
    # parser.add_argument('--outmap_dir', type=str, default="/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/mapblock/")
    # parser.add_argument('--out_dir', type=str, default="/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/test_docimgsplit811test_epoch73/block/")
    return parser.parse_args()


def main(cfg):
    stride = 512
    save_results_path = '/data1/zhengkengtao/exps/1424_DIDNet_docimgsplit811_crop512train_Cls_noAug/testepoch43_docimgsplit811_Orig200Psc200Mosaic200Ps200_crop512stride{}/'.format(stride)
    if (os.path.exists(save_results_path) == False): os.makedirs(save_results_path)
    cls_csv = save_results_path + 'crop512stride{}_cls.csv'.format(stride)

    # 图像块
    # filenames = glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_crop512stride128/*.png') + \
    #             glob.glob('/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride128/*.png')

    filenames = glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_crop512stride{}/*.png'.format(stride)) + \
                glob.glob('/data1/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride{}/*.png'.format(stride)) + \
                glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic_crop512stride{}/*.png'.format(stride)) + \
                glob.glob('/data1/zhengkengtao/docimg/docimg_split811/test_tamper_crop512stride{}/*.png'.format(stride))
    filenames.sort()
    print(len(filenames))

    # 整图
    image_name_list = glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig/*.jpg') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig/*.JPG') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/test_images/*.png') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic/*.png') + \
                      glob.glob('/data1/zhengkengtao/docimg/docimg_split811/test_tamper/*.png')
    image_name_list.sort()
    t1 = time.time()

    test_set = DataSetLoader(filenames)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=cfg.batch_size, shuffle=False)

    modelTest = DIDNet()
    modelTest = nn.DataParallel(modelTest)
    # pretrained_dict = torch.load('/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/DIDNet_epoch_86.pth')
    # pretrained_dict = torch.load('/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/DIDNet_epoch_73.pth')
    pretrained_dict = torch.load('/data1/zhengkengtao/exps/1424_DIDNet_docimgsplit811_crop512train_Cls_noAug/DIDNet_epoch_43.pth')

    modelTest.load_state_dict(pretrained_dict['state_dict'])

    accs = 0
    f = open(cls_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['name', 'gt', 'clspred'])
    modelTest.eval()
    with torch.no_grad():
        batch_idx = 0
        for data in tqdm(test_loader):
            image, image_name, srm = data
            image, srm = image.to(cfg.device), srm.to(cfg.device)
            # _, _, height, width = np.shape(image)
            output = modelTest(image, srm)  # shape: (b, 2)
            _, predict = torch.max(output.data, dim=1)
            predict_label = predict[0]
            if 'psc_' in image_name or 'ps_' in image_name:
                gt3name = image_name.replace('_images', '_gt3')
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
            acc = get_acc(clsgt, predict_label.cpu().data.numpy())
            accs += acc
            csv_writer.writerow([filenames[batch_idx], clsgt, int(predict_label.cpu().data.numpy())])
            batch_idx += 1
    acc = accs / len(test_loader)
    print('ACC: {:.5f}'.format(acc))
    csv_writer.writerow(['Patch Average Acc', len(test_loader), acc])
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


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)

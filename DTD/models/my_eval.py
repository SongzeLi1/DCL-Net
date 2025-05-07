import os
import cv2
import lmdb
import torch
# import jpegio
import numpy as np
import torch.nn as nn
import gc
import math
import time
import copy
import logging
import torch.optim as optim
import torch.distributed as dist
import pickle
import six
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss
import albumentations as A
from dtd import *
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
import tempfile
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

parser = argparse.ArgumentParser()
parser.add_argument('--pth', type=str, default='../pths/dtd_doctamper.pth')
args = parser.parse_args()


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
        gt3name = name.replace('tamper', 'masks')
        gt3name = gt3name.replace('psc_', 'gt3_')
        gt3 = Image.open(gt3name).convert('L')
        gt3 = np.array(gt3, np.uint8)
        gt3[gt3 == 255] = 0
        gt3[gt3 == 76] = 1
        gt3[gt3 == 29] = 1
        if self.transform:
            transformed = self.transform(image=image, mask=gt3)
            image = transformed['image']
            gt3 = transformed['mask']
        return {
            'image': image,
            'label': gt3,
        }


def train_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size[0] == input_size[1]:
        return A.Compose([
        A.Resize(input_size[0], input_size[0]),  # width, height
        A.Normalize(mean=normMean, std=normStd),
        ToTensorV2(),
    ])


train_names = glob('/pubdata/lisongze/docimg/exam/docimg_data/test/tamper/*.png')
train_names.sort()
test_data = DOCDataset(train_names, transform=train_transform([512, 512]))   # input_size[512, 512]


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

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

class IOUMetric:
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

model = seg_dtd('',2).cuda()
model = torch.nn.DataParallel(model)

def eval_net_dtd(model, test_data, plot=False,device='cuda'):
    train_loader1 = DataLoader(dataset=test_data, batch_size=6, num_workers=12, shuffle=False)
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    ckpt = torch.load(args.pth,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    iou=IOUMetric(2)
    precisons = []
    recalls = []
    with torch.no_grad():
        for batch_idx, batch_samples in enumerate(tqdm(train_loader1)):
            data, target, dct_coef, qs, q = batch_samples['image'], batch_samples['label'],batch_samples['rgb'], batch_samples['q'],batch_samples['i']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(target.to(device)), Variable(dct_coef.to(device)), Variable(qs.unsqueeze(1).to(device))
            pred = model(data,dct_coef,qs)               
            predt = pred.argmax(1)
            pred=pred.cpu().data.numpy()
            targt = target.squeeze(1)
            matched = (predt*targt).sum((1,2))
            pred_sum = predt.sum((1,2))
            target_sum = targt.sum((1,2))
            precisons.append((matched/(pred_sum+1e-8)).mean().item())
            recalls.append((matched/target_sum).mean().item())
            pred = np.argmax(pred,axis=1)
            iou.add_batch(pred,target.cpu().data.numpy())
        acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
        precisons = np.array(precisons).mean()
        recalls = np.array(recalls).mean()
        print('[val] iou:{} pre:{} rec:{} f1:{}'.format(iu,precisons,recalls,(2*precisons*recalls/(precisons+recalls+1e-8))))

eval_net_dtd(model, test_data)



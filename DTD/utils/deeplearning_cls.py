import os
import time
import copy
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.optim as optim

from glob import glob
from PIL import Image
from tqdm import tqdm
from .custom_lr import ShopeeScheduler
from .ranger import Ranger
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from utils.utils import AverageMeter, second2time, inial_logger
from .metric import IOUMetric
from torch.cuda.amp import autocast, GradScaler # need pytorch>1.6
from segmentation_models_pytorch.losses import DiceLoss,FocalLoss,SoftCrossEntropyLoss,LovaszLoss,SoftBCEWithLogitsLoss
from .metric import *
from .segmetric import *
import torch.nn.functional as F

#--mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size)#.cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def train_net(fold, logger, param, model, train_data, valid_data, device='cuda'):
    # 初始化参数
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    iter_inter      = param['iter_inter']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']
    save_epoch = param['save_epoch']
    T0 = param['T0']
    scaler = GradScaler()

    # logger.info(model)
    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4 ,weight_decay=5e-4)
    # optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-4) # 微调
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4) # rru
    # optimizer = Ranger(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-6, last_epoch=-1)
    # DiceLoss_fn = DiceLoss(mode='binary')
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    # DiceLoss_fn = DiceLoss(mode='multiclass')
    # FocalLoss_fn = FocalLoss(mode='multiclass')
    # LovaszLoss_fn = LovaszLoss(mode='binary')
    # LovaszLoss_fn = LovaszLoss(mode='multiclass')
    # SoftCrossEntropy_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.]).cuda()).cuda()

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    epoch_start = 0
    best_epoch = 0
    best_score = 0
    lowestloss_train_epoch = 0
    lowest_train_epoch_loss = 1e4
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Fold:{} Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(fold, epochs, x, y, train_data_size, valid_data_size))

    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_epoch_celoss = AverageMeter()
        train_iter_loss = AverageMeter()
        train_iter_acc = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, gt, dct, qs, clsgt = batch_samples['image'], batch_samples['label'], batch_samples['dct'], batch_samples[
                'qs'], batch_samples['clsgt']
            data, target, dct_coef, qs, clsgt = Variable(data.to(device)), Variable(gt.to(device)), Variable(
                dct.to(device)), Variable(qs.unsqueeze(1).to(device)), Variable(clsgt.to(device))

            clsgt = clsgt.to(torch.float32)  # [b,1]
            clsgt = clsgt.unsqueeze(0)
            with autocast(): #need pytorch > 1.6
                pred = model(data, dct_coef, qs) # [b,3,h,w]
                pred = pred.permute(1, 0)
                loss_ce = nn.BCEWithLogitsLoss().cuda()(pred, clsgt)

                pred = torch.sigmoid(pred)
                pred = pred > 0.5
                acc = torch.eq(pred, clsgt).sum().float().item()
                acc /= clsgt.shape[1]


                loss = loss_ce
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / train_loader_size) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_epoch_celoss.update(loss_ce.item())
            train_iter_loss.update(image_loss)
            train_iter_acc.update(acc)

            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] fold:{} epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    fold, epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'], train_iter_loss.avg,
                    spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()
                train_iter_acc.reset()
        # 保存train loss最低的模型
        if train_epoch_loss.avg < lowest_train_epoch_loss:
            torch.save(model.state_dict(), '{}/lowest_loss.pth'.format(save_ckpt_dir))
            lowest_train_epoch_loss = train_epoch_loss.avg
            lowestloss_train_epoch = epoch
            logger.info(
                'fold {} Lowest Train Loss Model saved at epoch:{} ~~~~~~~~~~~~~~~~~~~'.format(fold, epoch))
        logger.info(
            '[--lowest train loss--] lowestloss_epoch:{} lowestloss:{:.4f}'.format(lowestloss_train_epoch,
                                                                                   lowest_train_epoch_loss))

        # scheduler.step()
        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        valid_epoch_acc = AverageMeter()

        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                datas, gt3s, dct2, qs2, clsgts = batch_samples['image'], batch_samples['label'], batch_samples['dct'], \
                batch_samples['qs'], batch_samples['clsgt']
                datas, gt3s, dct_coef2, qs2, clsgts = Variable(datas.to(device)), Variable(gt3s.to(device)), Variable(
                    dct2.to(device)), Variable(qs2.unsqueeze(1).to(device)), Variable(clsgts.to(device))

                clsgts = clsgts.to(torch.float32)
                clsgts = clsgts.unsqueeze(0)

                cls_predicts = model(datas, dct2, qs2)

                cls_predicts = cls_predicts.permute(1, 0)
                val_ce_loss = nn.BCEWithLogitsLoss().cuda()(cls_predicts, clsgts)  # 图像块二分类loss

                cls_predicts = torch.sigmoid(cls_predicts)
                cls_predicts = cls_predicts > 0.5
                val_acc = torch.eq(cls_predicts, clsgts).sum().float().item()
                val_acc /= clsgts.shape[1]

                valloss = val_ce_loss
                image_loss = valloss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
                valid_epoch_acc.update(val_acc)

        score = valid_epoch_acc.avg
        logger.info('[val] fold:{} epoch:{} loss:{:.6f} val_acc:{:.6f}'.format(fold, epoch, valid_epoch_loss.avg,
                                                                               valid_epoch_acc.avg))

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # 保存最近模型
        torch.save(model.state_dict(), '{}/latest-ckpt.pth'.format(save_ckpt_dir))
        latest_ckpt_full_path = os.path.join(save_ckpt_dir, 'latest-ckpt-full.pth')
        torch.save(state, latest_ckpt_full_path)  # 含其它参数
        logger.info('fold {} latest model saved at epoch {}'.format(fold, epoch))
        # 保存余弦学习率模型
        if epoch in save_epoch[T0]:
            torch.save(model.state_dict(), '{}/cosine_fold_{}_epoch{}.pth'.format(save_ckpt_dir, fold, epoch))
        # 保存最优模型
        if score > best_score:
            torch.save(model.state_dict(), '{}/fold_{}_best-ckpt.pth'.format(save_ckpt_dir, fold))  # 不含其它参数
            filename = os.path.join(save_ckpt_dir, 'fold_{}_checkpoint-best.pth'.format(fold))
            torch.save(state, filename)  # 含其它参数
            best_score = score
            logger.info('fold {} Best Model saved at epoch:{} ============================='.format(fold, epoch))
            best_epoch = epoch
        logger.info('[--best val--] fold:{} best epoch:{} best score:{:.4f}'.format(fold, best_epoch, best_score))
        # # 间隔epoch保存模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), '{}/epoch{}.pth'.format(save_ckpt_dir, epoch))
        logger.info('epoch train+val time:{:.2f} min'.format((time.time() - epoch_start) / 60))
    return best_score


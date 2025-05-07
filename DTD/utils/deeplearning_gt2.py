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
    global latest_ckpt_full_path, state
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
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=8)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4) # dtdnet 训练
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)  # dtdnet 微调
    # optimizer = Ranger(model.parameters(),lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-6, last_epoch=-1)
    # DiceLoss_fn = DiceLoss(mode='binary')
    LovaszLoss_fn=LovaszLoss(mode='multiclass')
    SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
    # FocalLoss_fn_binary = FocalLoss(mode='binary')
    # DiceLoss_fn = DiceLoss(mode='multiclass')
    # FocalLoss_fn = FocalLoss(mode='multiclass')
    # LovaszLoss_fn = LovaszLoss(mode='binary')
    # LovaszLoss_fn = LovaszLoss(mode='multiclass')
    # SoftCrossEntropy_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.]).cuda()).cuda()

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_score = 0
    best_epoch = 0
    epoch_start = 0
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
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, gt, dct, qs = batch_samples['image'], batch_samples['label'],batch_samples['dct'], batch_samples['qs']
            data, target, dct_coef, qs = Variable(data.to(device)), Variable(gt.to(device)), Variable(dct.to(device)), Variable(qs.unsqueeze(1).to(device))

            gt = gt.to(torch.int64).to(device) # [b,h,w]
            # gt = gt.to(torch.float16)  # [b,h,w]
            with autocast(): #need pytorch > 1.6
                # # 1通道
                pred = model(data, dct_coef, qs).squeeze(1) # [b,1,h,w]->[b,h,w]
                # 3通道
                # pred = model(data) # [b,3,h,w]
                if param['mixup']:
                    data, targets_a, targets_b, lam = mixup_data(data, gt, 0.2)
                    outputs = model(data, dct_coef, qs).squeeze(1) # 将[b,1,h,w]->[b,h,w]
                    loss_lov = mixup_criterion(LovaszLoss_fn, outputs, targets_a, targets_b, lam)
                    loss_ce = mixup_criterion(SoftCrossEntropy_fn, outputs, targets_a, targets_b, lam)
                else:
                    # print(pred.shape, target.shape) # [b,c,h,w], [b,h,w]
                    # loss_dice = DiceLoss_fn(pred, target) # dice里面只对target做了onehot->[b,c,h,w] # dice里面多分类会做softmax，二分类会做sigmoid
                    # loss_focal = FocalLoss_fn(pred, gt)
                    loss_lov = LovaszLoss_fn(pred, gt)
                    loss_ce = SoftCrossEntropy_fn(pred, gt) # gt需要转化为float
                    # loss_focal_binary = FocalLoss_fn_binary(pred, gt)

                # loss = 3 * loss_dice + loss_ce
                # loss = loss_focal + loss_lovas
                # loss = loss_focal
                loss = loss_ce + loss_lov
                # loss = loss_focal_binary
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            # scheduler.step(epoch + batch_idx / train_loader_size)
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] fold:{} epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    fold,epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[0]['lr'],
                    train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        # scheduler.step()
        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()

        with torch.no_grad():
            # f1s, aucs, ious, mccs = [], [], [], []
            f1forgerys, iouforgerys = [], []
            for batch_idx, batch_samples in enumerate(valid_loader):
                data2, gt2s, dct2, qs2 = batch_samples['image'], batch_samples['label'], batch_samples['dct'], batch_samples['qs']
                data2, gt2s, dct_coef2, qs2 = Variable(data2.to(device)), Variable(gt2s.to(device)), Variable(
                    dct2.to(device)), Variable(qs2.unsqueeze(1).to(device))

                gt2s = gt2s.to(torch.int64).to(device)  # [b,h,w]
                predicts = model(data2, dct_coef2, qs2)
                # print(predicts.shape, predicts.min(), predicts.max())
                # out = torch.sigmoid(predicts)
                # print(out.shape, out.min(), out.max())
                predicts = predicts.argmax(1)
                # predicts = (torch.sigmoid(predicts) > 0.5).int().detach().cpu()
                predicts = predicts.cpu().data.numpy()
                gt2s = gt2s.cpu().data.numpy()
                # out = out.squeeze(1).float().cpu().data.numpy()
                # print(gt2s.shape, gt2s.min(), gt2s.max()) # 0 1
                # print(predicts.shape, predicts.min(), predicts.max())
                f1forgery, iouforgery = get_f1_iou(gt2s, predicts)
                f1forgerys.append(f1forgery)
                iouforgerys.append(iouforgery)

                # predicts = F.softmax(predicts, dim=1)
                # predicts = predicts.cpu().data.numpy() # [b,c,h,w]
                # predicts = np.argmax(predicts, axis=1) # [b,h,w]
                # # print(predicts.shape, predicts.min(), predicts.max())
                # gt2s = gt2s.cpu().data.numpy()
                # f1forgery, iouforgery = get_f1_iou(gt2s, predicts)
                # f1forgerys.append(f1forgery)
                # iouforgerys.append(iouforgery)

                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)

        # f1_mean, auc_mean, iou_mean, mcc_mean = np.mean(f1s), np.mean(aucs), np.mean(ious), np.mean(mccs)
        # logger.info('[val] epoch:{} f1:{} iou:{} auc:{} mcc:{}'.format(epoch, f1_mean, iou_mean, auc_mean, mcc_mean))
        # score = f1_mean + iou_mean
        # logger.info('[val] fold:{} epoch:{} f1:{:.3f} iou:{:.3f} score:{:.3}'.format(fold, epoch, f1_mean, iou_mean, score))
        # logger.info('[val] fold:{} best epoch:{} best score:{}'.format(fold, best_epoch, best_score))

        f1forgery_mean, iouforgery_mean = np.mean(f1forgerys), np.mean(iouforgerys)
        score = f1forgery_mean
        logger.info('[val] fold:{} epoch:{} f1foregry:{:.4f} iouforgey:{:.4f} score:{:.4f}'
                    .format(fold, epoch, f1forgery_mean, iouforgery_mean, score))

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
            torch.save(model.state_dict(),'{}/cosine_fold_{}_epoch{}.pth'.format(save_ckpt_dir,fold,epoch))
        # 保存最优模型
        if score > best_score:  # train_loss_per_epoch valid_loss_per_epoch
            torch.save(model.state_dict(), '{}/fold_{}_best-ckpt.pth'.format(save_ckpt_dir, fold))  # 不含其它参数
            filename = os.path.join(save_ckpt_dir,'fold_{}_checkpoint-best.pth'.format(fold) )
            torch.save(state, filename)
            best_score = score
            logger.info('fold {} Best Model saved at epoch:{} ============================='.format(fold, epoch))
            best_epoch = epoch
        logger.info('[--best val--] fold:{} best epoch:{} best score:{:.4f}'.format(fold, best_epoch, best_score))
        # 间隔epoch保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), '{}/epoch{}.pth'.format(save_ckpt_dir, epoch))
        logger.info('epoch train+val time:{:.2f} min'.format((time.time() - epoch_start) / 60))
    return best_score


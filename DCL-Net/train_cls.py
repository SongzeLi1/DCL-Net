import os
from utils.utils import AverageMeter, inial_logger
from torch.utils.data import Dataset,DataLoader
from contdnet.contdnet_cls import *
import warnings
import glob
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler # need pytorch>1.6
from utils.metric import *
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import time


warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        h, w = image.shape[0], image.shape[1]
        if 'orig' in name:
            clsgt = 0
            gt3 = np.zeros((h, w), np.uint8)
        else:
            gt3name = name.replace('_images', '_gt3')
            gt3name = gt3name.replace('psc_', 'gt3_')
            gt3 = Image.open(gt3name).convert('L')
            gt3 = np.array(gt3, np.uint8)
            gt3[gt3 == 255] = 0
            gt3[gt3 == 76] = 1
            gt3[gt3 == 29] = 2
            if 1 in gt3:
                clsgt = 1
            else:
                clsgt = 0
        if self.transform:
            transformed = self.transform(image=image, mask=gt3)
            image = transformed['image']
            gt3 = transformed['mask']
        return {
            'image': image,
            'label': gt3,
            'clsgt': clsgt
        }

def contra(featcon, gt01con, contrast_temperature=0.07):
    msk = torch.eq(gt01con, gt01con.transpose(1, 2)).float()
    mskneg = torch.add(torch.negative(msk), 1)
    featcon_logits = torch.div(torch.matmul(featcon, featcon.transpose(1, 2)), contrast_temperature)
    flag = torch.eye(featcon_logits.shape[-1]).cuda()
    neg_flag = torch.add(torch.negative(flag), 1).detach()
    featcon_logits = torch.mul(featcon_logits, neg_flag)
    featcon_logits_max, _ = torch.max(featcon_logits, dim=1, keepdim=True)
    featcon_logits = featcon_logits - featcon_logits_max.detach()
    featcon_logits = torch.exp(featcon_logits)
    neg_sum = torch.sum(torch.mul(featcon_logits, mskneg), dim=-1)
    denominator = torch.add(featcon_logits, neg_sum.unsqueeze(dim=-1))
    division = torch.div(featcon_logits, denominator + 1e-18)
    loss_matrix = -torch.log(division + 1e-18)
    loss_matrix = torch.mul(loss_matrix, msk)
    loss_matrix = torch.mul(loss_matrix, neg_flag)
    loss = torch.sum(loss_matrix, dim=-1)
    loss = torch.div(loss, msk.sum(dim=-1) - 1 + 1e-18)
    return loss

def train_net(fold, logger, param, model, train_data, valid_data):
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, T_mult=2, eta_min=1e-6, last_epoch=-1)

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
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
        loss_weight = 0.1
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_epoch_celoss = AverageMeter()
        train_epoch_contraloss = AverageMeter()
        train_iter_loss = AverageMeter()
        train_iter_acc = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, gt, clsgt = batch_samples['image'].cuda(), batch_samples['label'].cuda(), batch_samples['clsgt'].cuda()
            gt = gt.to(torch.float32)  # [b,h,w]
            gt01 = gt.clone()
            gt01[gt01 == 2] = 0
            clsgt = clsgt.to(torch.float32)  # [b,1]
            clsgt = clsgt.unsqueeze(0)
            with autocast(): # need pytorch > 1.6
                # 3通道
                cls_pred, contra_pred = model(data)  # [b,3,h,w] [b,2,h,w]
                cls_pred = cls_pred.permute(1, 0)
                ce_loss = nn.BCEWithLogitsLoss().cuda()(cls_pred, clsgt)  # 图像块二分类loss

                cls_pred = torch.sigmoid(cls_pred)
                cls_pred = cls_pred > 0.5
                acc = torch.eq(cls_pred, clsgt).sum().float().item()
                acc /= clsgt.shape[1]

                # -----计算contrastive loss-------
                featcon = contra_pred
                Ba, Ch, _, _ = featcon.shape
                featcon = featcon.view(Ba, Ch, -1)
                featcon = torch.transpose(featcon, 1, 2)
                featcon = F.normalize(featcon, dim=-1)

                gt01con = gt01.detach()
                gt01con = F.avg_pool2d(gt01con, kernel_size=32, stride=32)
                gt01con = (gt01con > 0.5).int().float()
                gt01con = gt01con.view(Ba, -1)
                gt01con = gt01con.unsqueeze(dim=1)

                contrast_temperature = 0.07  # MoCo=0.07, SimCLR=0.5
                contra_loss = contra(featcon, gt01con, contrast_temperature)
                contra_loss = contra_loss.mean(dim=-1)
                contra_loss = loss_weight * contra_loss.mean()
                # ------------------------------

                loss = ce_loss + contra_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            scheduler.step(epoch + batch_idx / train_loader_size)
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_epoch_celoss.update(ce_loss.item())
            train_epoch_contraloss.update(contra_loss.item())
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
            logger.info('fold {} Lowest Train Loss Model saved at epoch:{} ~~~~~~~~~~~~~~~~~~~'.format(fold, epoch))
        logger.info('[--lowest train loss--] lowestloss_epoch:{} lowestloss:{:.4f}'.format(lowestloss_train_epoch, lowest_train_epoch_loss))

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        valid_epoch_acc = AverageMeter()
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                datas, _, clsgts = batch_samples['image'].cuda(), batch_samples['label'].cuda(), batch_samples['clsgt'].cuda()
                clsgts = clsgts.to(torch.float32)  # [b,h,w]
                clsgts = clsgts.unsqueeze(0)

                cls_predicts, contra_predicts = model(datas)

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
        logger.info('[val] fold:{} epoch:{} loss:{:.6f} val_acc:{:.6f}'.format(fold, epoch, valid_epoch_loss.avg, valid_epoch_acc.avg))

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
        if score > best_score:
            torch.save(model.state_dict(), '{}/fold_{}_best-ckpt.pth'.format(save_ckpt_dir, fold)) # 不含其它参数
            filename = os.path.join(save_ckpt_dir,'fold_{}_checkpoint-best.pth'.format(fold) )
            torch.save(state, filename) # 含其它参数
            best_score = score
            logger.info('fold {} Best Model saved at epoch:{} ============================='.format(fold, epoch))
            best_epoch = epoch
        logger.info('[--best val--] fold:{} best epoch:{} best score:{:.4f}'.format(fold, best_epoch, best_score))
        # # 间隔epoch保存模型
        if epoch % 1 == 0:
            torch.save(model.state_dict(), '{}/epoch{}.pth'.format(save_ckpt_dir, epoch))
        logger.info('epoch train+val time:{:.2f} min'.format((time.time()-epoch_start)/60))
    return best_score

# No augmentation
def train_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size[0] == input_size[1]:
        return A.Compose([
        A.Resize(input_size[0], input_size[0]),  # width, height
        A.Normalize(mean=normMean, std=normStd),
        ToTensorV2(),
    ])


def val_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    return A.Compose([
    A.Resize(input_size[0], input_size[1]),
    A.Normalize(mean = normMean, std = normStd),
    ToTensorV2(),
])



if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 参数设置
    param = {}
    param['n_class'] = 3 # 2 3
    param['mixup'] = False
    param['input_size'] = [512, 512]
    param['epochs'] = 500
    param['batch_size'] = 22
    param['iter_inter'] = 80
    param['backbone'] = 'convnext'
    param['model_name'] = 'contdnet'
    param['save_train_dir'] = '/data1/1/exps/'
    param['multifold_train'] = False
    save_ckpt_dir = os.path.join(param['save_train_dir'], param['model_name'] + '_{}x{}'.format(param['input_size'][1], param['input_size'][0])) # width, height
    param['pretrain_ckpt'] = None
    param['load_ckpt_dir'] = None
    param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
    param['T0'] = 3  # cosine warmup的参数
    param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92,191]}

    n_class = param['n_class']
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    logger = inial_logger(os.path.join(save_ckpt_dir,'log.log'))
    logger.info(param)

    train_names = glob.glob('/data1/1/docimg/docimg_split811/crop512x512/train_images/*.png') + \
                  glob.glob('/data1/1/docimg/docimg_split811/train_images_orig_crop512stride512/*.png')
    train_names.sort()
    val_names = glob.glob('/data1/1/docimg/docimg_split811/crop512x512/val_images/*.png') + \
                glob.glob('/data1/1/docimg/docimg_split811/val_images_orig_crop512stride512/*.png')
    val_names.sort()

    logger.info('Training started')
    logger.info("train: {} valid: {}".format(len(train_names), len(val_names)))

    train_data = DOCDataset(train_names, transform=train_transform(param['input_size']))
    valid_data = DOCDataset(val_names, transform=val_transform(param['input_size']))
    logger.info(train_transform(param['input_size']))
    model = ConTDNet_cls()
    model = torch.nn.DataParallel(model)
    model.cuda()
    if param['pretrain_ckpt'] is not None:
        model.load_state_dict(torch.load(param['pretrain_ckpt']))
    score = train_net(0, logger, param, model, train_data, valid_data)









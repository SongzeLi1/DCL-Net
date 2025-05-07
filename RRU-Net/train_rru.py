import os
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
import torch
import numpy as np
# from dataset.DOCDataset import DOCDataset
from dataset.DOCDataset_gt2s import DOCDataset
from dataset.transform import train_transform, val_transform
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from utils.utils import AverageMeter, second2time, inial_logger
from PIL import Image, ImageFile
from torch.utils.data import Dataset,DataLoader
# from networks.seg_qyl.seg_qyl import * # seg_qyl
from networks.rrunet.unet_model import *
# from networks.difnet.difnet import *
# from utils.deeplearning import *
from utils.deeplearning_gt2 import *
import warnings

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# conda activate pt
# cd /home/zhengkengtao/codes/docimg_forensics/
# python train_rru.py
# 参数设置
param = {}
param['n_class'] = 1
param['mixup'] = False
param['input_size'] = [512, 512] # width, height [1664, 1664], [1440, 1440] [1024, 1024]
param['epochs'] = 200        # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['batch_size'] = 8*7      # 批大小
param['disp_inter'] = 1      # 显示间隔(epoch)
param['iter_inter'] = 40     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['backbone'] = 'rru'        # efficientnet-b2 se_resnext101_32x4d efficientnet-b7 convnext se_resnet50
param['model_name'] = 'rru'         # 模型名称 DeepLabV3Plus UnetPlusPlus difnet
param['save_train_dir'] = '/pubdata/zhengkengtao/exps/1023_RRUNet_docimg_split811_png_Aug_OrigMosaicOneKind_crop512x512/'
param['multifold_train'] = False
save_ckpt_dir = os.path.join(param['save_train_dir'], param['model_name'] + '_{}x{}'.format(param['input_size'][1], param['input_size'][0])) # width, height
param['pretrain_ckpt'] = None
param['load_ckpt_dir'] = '/pubdata/zhengkengtao/exps/1023_RRUNet_docimg_split811_png_Aug_OrigMosaicOneKind_crop512x512/rru_512x512/latest-ckpt-full.pth'
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0'] = 3  # cosine warmup的参数
param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92,191]}

n_class = param['n_class']
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
logger = inial_logger(os.path.join(save_ckpt_dir,'log.log'))
logger.info(param)

# 准备数据集
# 已经分好train和val数据进行一次普通训练
if param['multifold_train'] == False:

    # data_dir = '/pubdata/zhengkengtao/docimg/docimg_split811/'
    # train_imgs_dir = os.path.join(data_dir, "train_images/")
    # train_labels_dir = os.path.join(data_dir, "train_gt3/")
    # # train_labels_dir = os.path.join(data_dir, "train_labels/")
    # val_imgs_dir = os.path.join(data_dir, "val_images/") # val_images
    # val_labels_dir = os.path.join(data_dir, "val_gt3/") # val_gt3

    data_dir = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/patch_noblank/'
    train_imgs_dir = os.path.join(data_dir, "train_images/")
    train_labels_dir = os.path.join(data_dir, "train_gt3/")
    # train_labels_dir = os.path.join(data_dir, "train_labels/")
    val_imgs_dir = os.path.join(data_dir, "val_images/") # val_images
    val_labels_dir = os.path.join(data_dir, "val_gt3/") # val_gt3

    # data_dir = "/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split811/"
    # val_labels_dir = os.path.join(data_dir, "val_labels/")
    # train_imgs_dir = os.path.join(data_dir, "train_imgs/")
    # train_labels_dir = os.path.join(data_dir, "train_gt/")
    # val_imgs_dir = os.path.join(data_dir, "val_imgs/")
    # val_labels_dir = os.path.join(data_dir, "val_gt/")
    # val_labels_dir = os.path.join(data_dir, "val_labels/")

    train_names = os.listdir(train_imgs_dir)
    val_names = os.listdir(val_imgs_dir)
    train_names.sort()
    val_names.sort()
    # train_names = train_names[:56]
    # val_names = val_names[:28]
    # train_names = train_names[0::4]
    # val_names = val_names[0::3]

    logger.info('Training started')
    logger.info("train: {} valid: {}".format(len(train_names), len(val_names)))
    train_data = DOCDataset(train_names, train_imgs_dir, train_labels_dir, transform=train_transform(param['input_size']))
    valid_data = DOCDataset(val_names, val_imgs_dir, val_labels_dir, transform=val_transform(param['input_size']))
    # model = seg_qyl(param['backbone'], param['model_name'], n_class)
    # model = DIF()
    model = Ringed_Res_Unet(n_channels=3, n_classes=1)
    model = torch.nn.DataParallel(model)
    model.cuda()
    if param['pretrain_ckpt'] is not None:
        model.load_state_dict(torch.load(param['pretrain_ckpt']))
    score = train_net(0, logger, param, model, train_data, valid_data)
# 多折训练验证
else:
    data_dir = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_val_split_91/train_val_images/'
    labels_dir = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_val_split_91/train_val_labels/'
    files = os.listdir(data_dir)
    img_names = []
    for img in files:
        img_names.append(img.split('.')[0])
    # 将训练数据拆分训练集和验证集
    folds = KFold(n_splits=5, shuffle=True, random_state=42).split(np.arange(len(img_names)))  # 将数据分成n_splits份
    acc_lst = []
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:  # 定几折
            break
        logger.info('Training with Fold {} started'.format(fold))
        logger.info("train: {} valid: {}".format(len(trn_idx), len(val_idx)))
        train_names = [img_names[i] for i in trn_idx]
        val_names = [img_names[i] for i in val_idx]
        train_data = DOCDataset(train_names, data_dir, labels_dir, transform=train_transform(param['input_size']))
        valid_data = DOCDataset(val_names, data_dir, labels_dir, transform=val_transform(param['input_size']))
        model = seg_qyl(param['backbone'], param['model_name'], n_class).cuda()
        model = torch.nn.DataParallel(model)
        # model.load_state_dict(torch.load(''))
        score = train_net(fold, logger, param, model, train_data, valid_data)
        acc_lst.append(score)





import os
import torch
import numpy as np
# from dataset.DOCDataset import DOCDataset
from dataset.DOCDataset_gt2s_type3 import DOCDataset
from dataset.transform import train_transform, val_transform
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from utils.utils import AverageMeter, second2time, inial_logger
from PIL import Image, ImageFile
from torch.utils.data import Dataset,DataLoader
# from networks.seg_qyl.seg_qyl import * # seg_qyl
from dtd import *
# from networks.rrunet.unet_model import *
# from networks.difnet.difnet import *
# from utils.deeplearning import *
from utils.deeplearning_gt2_type3 import *
import warnings

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_type2.py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6"
# conda activate pt
# cd /home/zhengkengtao/codes/docimg_forensics/
# python train_rru.py
# 参数设置
param = {}
param['mixup'] = False
param['input_size'] = [512, 512] # width, height [1664, 1664], [1440, 1440] [1024, 1024]
param['epochs'] = 200        # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['batch_size'] = 8*7      # 批大小
param['disp_inter'] = 1      # 显示间隔(epoch)
param['iter_inter'] = 40     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['backbone'] = 'resnet18'        # efficientnet-b2 se_resnext101_32x4d efficientnet-b7 convnext se_resnet50
param['model_name'] = 'dtd'         # 模型名称 DeepLabV3Plus UnetPlusPlus difnet
param['save_train_dir'] = '/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type3/'
param['multifold_train'] = False
save_ckpt_dir = os.path.join(param['save_train_dir'], param['model_name'] + '_{}x{}'.format(param['input_size'][1], param['input_size'][0])) # width, height
param['pretrain_ckpt'] = None
param['load_ckpt_dir'] = None   #'/pubdata/zhengkengtao/exps/1023_RRUNet_docimg_split811_png_Aug_OrigMosaicOneKind_crop512x512/rru_512x512/latest-ckpt-full.pth'
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0'] = 3  # cosine warmup的参数
param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92,191]}

if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
logger = inial_logger(os.path.join(save_ckpt_dir,'log.log'))
logger.info(param)

# 准备数据集
data_dir = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/patch_noblank/'
train_imgs_dir = '/pubdata/lisongze/docimg/exam/docimg2jpeg/train_images_75_100/'
# train_imgs_dir = os.path.join(data_dir, "train_images/")
train_labels_dir = os.path.join(data_dir, "train_gt3/")
val_imgs_dir = '/pubdata/lisongze/docimg/exam/docimg2jpeg/val_images_75_100/'
# val_imgs_dir = os.path.join(data_dir, "val_images/") # val_images
val_labels_dir = os.path.join(data_dir, "val_gt3/") # val_gt3

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
model = seg_dtd('', 2)  # 'backbone' num_class
model = torch.nn.DataParallel(model)
model.cuda()
if param['pretrain_ckpt'] is not None:
    model.load_state_dict(torch.load(param['pretrain_ckpt']))
score = train_net(0, logger, param, model, train_data, valid_data)





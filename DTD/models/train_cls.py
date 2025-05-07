import os
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_type2.py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import torch
import numpy as np
# from dataset.DOCDataset import DOCDataset
from dataset.DOCDataset_cls import DOCDataset
from dataset.transform import train_transform, val_transform
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from utils.utils import AverageMeter, second2time, inial_logger
from PIL import Image, ImageFile
from torch.utils.data import Dataset,DataLoader
# from networks.seg_qyl.seg_qyl import * # seg_qyl
from dtd_cls import *
# from networks.rrunet.unet_model import *
# from networks.difnet.difnet import *
# from utils.deeplearning import *
from utils.deeplearning_cls import *
import warnings
import glob

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
param['batch_size'] = 4*8      # 批大小
param['disp_inter'] = 1      # 显示间隔(epoch)
param['iter_inter'] = 300     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['backbone'] = 'resnet18'        # efficientnet-b2 se_resnext101_32x4d efficientnet-b7 convnext se_resnet50
param['model_name'] = 'dtd'         # 模型名称 DeepLabV3Plus UnetPlusPlus difnet
param['save_train_dir'] = '/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/cls_2class_102/'
param['multifold_train'] = False
save_ckpt_dir = os.path.join(param['save_train_dir'], param['model_name'] + '_{}x{}'.format(param['input_size'][1], param['input_size'][0])) # width, height
param['pretrain_ckpt'] = None   #"/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type1/dtd_512x512/latest-ckpt.pth"
param['load_ckpt_dir'] = None   #'/pubdata/zhengkengtao/exps/1023_RRUNet_docimg_split811_png_Aug_OrigMosaicOneKind_crop512x512/rru_512x512/latest-ckpt-full.pth'
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0'] = 3  # cosine warmup的参数
param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92,191]}

n_class = param['n_class']
if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
logger = inial_logger(os.path.join(save_ckpt_dir,'log.log'))
logger.info(param)

# 准备数据集
train_names = glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/train_images_75_100/*.jpg') + \
              glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/train_images_orig_crop512stride512_75_100/*.jpg')
train_names.sort()
val_names = glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/val_images_75_100/*.jpg') + \
            glob.glob('/pubdata/lisongze/docimg/exam/docimg2jpeg/val_images_orig_crop512stride512_75_100/*.jpg')
val_names.sort()

logger.info('Training started')
logger.info("train: {} valid: {}".format(len(train_names), len(val_names)))
train_data = DOCDataset(train_names, transform=train_transform(param['input_size']))
valid_data = DOCDataset(val_names, transform=val_transform(param['input_size']))
model = seg_dtd()  # 'backbone' num_class
model = torch.nn.DataParallel(model)
model.cuda()
if param['pretrain_ckpt'] is not None:
    model.load_state_dict(torch.load(param['pretrain_ckpt']))
score = train_net(0, logger, param, model, train_data, valid_data)





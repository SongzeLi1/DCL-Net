# conda activate pt
# cd /home/zhengkengtao/codes/docimg_forensics/
# python train_densefcn_nclass3.py

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from networks.denseFCN.denseFCN_nclass3 import *
from futils.deeplearning_densefcn_nclass3 import *
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# def train_transform(input_size):
#     normMean = [0.485, 0.456, 0.406]
#     normStd = [0.229, 0.224, 0.225]
#     if input_size[0] == input_size[1]:
#         return A.Compose([
#         A.Resize(input_size[0], input_size[0]),  # width, height
#         # A.ImageCompression(quality_lower=70, quality_upper=100, p=1), # 测png的时候不加压缩去训练
#         A.Normalize(mean=normMean, std=normStd),
#         ToTensorV2(),
#     ])

# augmentation
def train_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    return A.Compose([
    A.Resize(input_size[0], input_size[0]), # width, height
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
    ]),
    A.MedianBlur(p=0.3),
    A.GaussNoise(var_limit=(25.0, 900.0), mean=0, p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
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


class DOCDataset(Dataset):
    def __init__(self, names, imgs_dir, masks_dir, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.names = names
        logging.info(f'Creating dataset with {len(self.names)} examples')
    def __len__(self):
        return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        image = Image.open(self.imgs_dir + name).convert('RGB')
        if self.masks_dir is not None:
            gtname = name.replace('psc_', 'gt3_')
            gtname = gtname.replace('.jpg', '.png')
            gtname = gtname.replace('.tif', '.png')
            gtname = gtname.replace('_qf60', '')
            gtname = gtname.replace('_qf70', '')
            gtname = gtname.replace('_qf80', '')
            gtname = gtname.replace('_qf90', '')
            mask = Image.open(self.masks_dir + gtname).convert('L')
        else:
            mask = Image.open(self.imgs_dir + name).convert('RGB').convert('L')
        image = np.array(image, np.uint8)
        mask = np.array(mask)

        # 二分类
        # if mask.max() > 1: mask = np.uint8(mask / 255)
        # ---三分类 docimg---
        mask[mask == 255] = 0
        mask[mask == 76] = 1
        mask[mask == 29] = 2
        # # ---二分类 docimg Tamper和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 1
        # # ---二分类 docimg Orig和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 0
        # # ---二分类 Alinew, SUPATLANTIQUE---
        # mask[mask !=0] = 1
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': mask
        }


if __name__ == '__main__':
    # 参数设置
    param = {}
    param['n_class'] = 3
    param['mixup'] = False
    param['input_size'] = [512, 512] # width, height
    param['epochs'] = 500        # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
    param['batch_size'] = 20*4      # 批大小
    param['disp_inter'] = 1      # 显示间隔(epoch)
    param['iter_inter'] = 30     # 显示迭代间隔(batch)
    param['min_inter'] = 10
    param['backbone'] = 'densefcn'        # efficientnet-b2 se_resnext101_32x4d efficientnet-b7 convnext rru
    param['model_name'] = 'densefcn'         # 模型名称 DeepLabV3Plus UnetPlusPlus difnet rru
    param['save_train_dir'] = '/pubdata/zhengkengtao/exps/0730_denseFCN-nclass3_docimg_split811_png_AdamWFocalloss_Aug/'
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

    # 准备数据集
    # 已经分好train和val数据进行一次普通训练
    if param['multifold_train'] == False:
        # data_dir = "/data1/zhengkengtao/docimg/docimg_split811/"
        data_dir = "/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/patch_noblank/"
        train_imgs_dir = os.path.join(data_dir, "train_images/")
        train_labels_dir = os.path.join(data_dir, "train_gt3/")
        val_imgs_dir = os.path.join(data_dir, "val_images/")
        val_labels_dir = os.path.join(data_dir, "val_gt3/")
        # data_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/'
        # data_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/'
        # train_imgs_dir = os.path.join(data_dir, "train_imgs/")
        # train_labels_dir = os.path.join(data_dir, "train_gt/")
        # val_imgs_dir = os.path.join(data_dir, "val_imgs/")
        # val_labels_dir = os.path.join(data_dir, "val_gt/")
        train_names = os.listdir(train_imgs_dir)
        val_names = os.listdir(val_imgs_dir)
        train_names.sort()
        val_names.sort()
        # train_names = train_names[:8*4*2]
        # val_names = val_names[:8*4*2]
        logger.info('Training started')
        logger.info("train: {} valid: {}".format(len(train_names), len(val_names)))
        train_data = DOCDataset(train_names, train_imgs_dir, train_labels_dir, transform=train_transform(param['input_size']))
        valid_data = DOCDataset(val_names, val_imgs_dir, val_labels_dir, transform=val_transform(param['input_size']))
        logger.info(train_transform(param['input_size']))
        model = normal_denseFCN(bn_in='bn')
        print(model)
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





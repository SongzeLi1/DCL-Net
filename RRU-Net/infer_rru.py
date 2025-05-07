from utils.utils import inial_logger
from scipy.io import loadmat
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import colorEncode
from torch.utils.data import Dataset, DataLoader
import warnings
from networks.rrunet.unet_model import *
import torch
import cv2
from PIL import Image, ImageFile
from utils.metric import *
import time
import logging
Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        # # ---三分类 docimg---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 2
        # # ---二分类 docimg Tamper和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 1
        # # ---二分类 docimg Orig和Mosaic一类---
        # mask[mask == 255] = 0
        # mask[mask == 76] = 1
        # mask[mask == 29] = 0
        # ---二分类 Alinew, SUPATLANTIQUE---
        mask[mask !=0] = 1
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': mask
        }


if __name__=="__main__":
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4, 5, 6, 7'
    warnings.filterwarnings('ignore')
    n_class = 1
    mask_threshold, viz, save_pmap, save_mask, batch_size = 0.5, True, True, True, 6*6
    input_size = [768, 768]  # None [1792, 1792] [1024, 1024], [1440, 1440], [768, 768], [512, 512]
    patch_size, patch_stride = None, None
    tta_scale = None  # [128, 192, 256]
    # test_img_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
    # test_gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/' # None '/pubdata/zhengkengtao/docimg/docimg_split811/test_gt3/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split811/test_gt/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/img/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/mask/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_gt/'
    # test_img_path = '/data1/zhengkengtao/docimg/tamper_mosaic/'
    # test_gt_path = '/data1/zhengkengtao/docimg/gt3/'
    # test_img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split_1000_200_2800/test_imgs/'
    # test_gt_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split_1000_200_2800/test_gt/'
    test_img_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split118/test_imgs/'
    test_gt_path = '/pubdata/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split118/test_gt/'
    checkpoint_dir = '/pubdata/zhengkengtao/exps/1617_RRUNet_Alinew_trainsplit118_noAug_pretrainwithdocimgsplit811Aug/rru_768x768/fold_0_best-ckpt.pth'
    save_results_path = '/pubdata/zhengkengtao/exps/1617_RRUNet_Alinew_trainsplit118_noAug_pretrainwithdocimgsplit811Aug/rru_768x768/test_Alinew3200/'
    # save_results_path = '/data1/zhengkengtao/exps/0716_RRUNet_docimg_split811_png_Aug/rru_1024x1024/test_bestepoch86/pred_noresize/'
    pmap_path = save_results_path + 'unthreshold/'
    pred_path = save_results_path + 'threshold_{}/'.format(mask_threshold)
    if (os.path.exists(pmap_path) == False): os.makedirs(pmap_path)
    if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
    logger = inial_logger(os.path.join(save_results_path, 'test.log'))
    filenames = os.listdir(test_img_path)
    filenames.sort()
    filenames = filenames
    mious, f1forgerys, iouforgerys, f1mosaics, ioumosaics = [], [], [], [], []
    t1 = time.time()
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_pmap:{}, save_mask:{}, batch_size:{}'.format(n_class,mask_threshold, viz, save_pmap, save_mask, batch_size))
    logger.info('test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames), input_size, patch_size, patch_stride, tta_scale))
    logger.info('checkpoint:{}'.format(checkpoint_dir))
    logger.info('test_img_path:{}'.format(test_img_path))
    logger.info('test_gt_path:{}'.format(test_gt_path))
    logger.info('save_test_results_path:{}'.format(save_results_path))
    logger.info('======================================================================================================')
    model = Ringed_Res_Unet(n_channels=3, n_classes=1)
    model.cuda()
    model = torch.nn.DataParallel(model)
    checkpoints = torch.load(checkpoint_dir)
    if 'state_dict' in checkpoints.keys():
        model.load_state_dict(checkpoints['state_dict'])
    else:
        model.load_state_dict(checkpoints)
    model.eval()
    test_data = DOCDataset(filenames, test_img_path, test_gt_path, infer_transform(input_size))
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    out_lst = []
    print('Total Test Batches: {}'.format(len(test_loader)))
    for batch_idx, batch_samples in enumerate(test_loader):
        img = batch_samples['image'].cuda()
        print('Testing batch {} ......'.format(batch_idx))
        with torch.no_grad():
            out = model(img)
        # 二分类
        # preds = (torch.sigmoid(out) > mask_threshold).squeeze().int().detach().cpu().numpy()
        out = torch.sigmoid(out).squeeze().cpu().numpy()
        # print(out.shape) # [b,h,w]
        # # # 三分类
        # out = F.softmax(out, dim=1)  # [b,c,h,w]
        # out = out.cpu().data.numpy() # [b,c,h,w]
        # preds = np.argmax(out, axis=1)  # [b,h,w]
        # print(preds.shape,preds.min(),preds.max()) # [b,h,w], 0-2
        for i in range(out.shape[0]):
            out_lst.append(out[i])
    # np.save(save_results_path + 'pmap_results.npy', np.array(out_lst)) # 将测试结果保存为npy
    # out_lst = np.load(save_results_path + 'pmap_results.npy')
    for i in range(len(filenames)):
        out = out_lst[i]
        img_name = filenames[i]
        print(img_name)
        image = cv2.imread(test_img_path + img_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ht, wd, _ = image.shape
        if input_size is not None:
            out = cv2.resize(out, (wd, ht), interpolation=cv2.INTER_NEAREST)
        # print(pred.shape, pred.min(), pred.max()) # [h,w] 0, 2
        # pred = np.int64(pred)
        if save_pmap:
            pmap = Image.fromarray(np.uint8(out * 255))
            pmap = pmap.convert('L')
            pmap.save(pmap_path + img_name[:-4] + '.png')
        if save_mask:
            pred = np.array(out>mask_threshold, dtype=np.uint8)
            pred = Image.fromarray(np.uint8(pred * 255))
            pred = pred.convert('L')
            pred.save(pred_path + img_name[:-4] + '.png')
    print('test time:', time.time()-t1)

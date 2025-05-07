from utils.utils import inial_logger
from scipy.io import loadmat
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import colorEncode
from torch.utils.data import Dataset, DataLoader
from networks.mvss_net.mvssnet_nclass3 import *
from utils.segmetric import *
import torch
import torch.nn.functional as F
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

        # print(np.array(image).min(), np.array(image).max()) # 0, 255
        # print(np.array(mask).min(), np.array(mask).max()) # 0, 1
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return {
            'image': image,
            'label': mask
        }


def visualize_result(img_dir, pred):
    img = cv2.imread(img_dir, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 1023
    colors = loadmat('demo/color150.mat')['colors']
    names = {
            0: "背景",
            1: "篡改",
        }
    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    #
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx]]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.001:
            print("  {}: {:.2f}%".format(name, ratio))
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint16)
    # aggregate images and save
    #print(pred_color.shape)
    pred_color=cv2.resize(pred_color,(img.shape[1],img.shape[0]))
    #im_vis = np.concatenate((img, pred_color), axis=1)
    return pred_color

def tta(img, model):
    with torch.no_grad():
        img = img.cuda()
        out1 = model(img)
        out2 = model(torch.flip(img, dims=[2]))
        out2 = torch.flip(out2, dims=[2])
        out3 = model(torch.flip(img, dims=[3]))
        out3 = torch.flip(out3, dims=[3])
        out = (out1 + out2 + out3) / 3.0
    return out


def get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, n_input=3, constrain=True, **kwargs):
    model = MVSSNet(nclass, backbone=backbone,
                    pretrained_base=pretrained_base,
                    sobel=sobel,
                    n_input=n_input,
                    constrain=constrain,
                    **kwargs)
    return model


if __name__=="__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    warnings.filterwarnings('ignore')
    n_class = 3
    mask_threshold, viz, save_map, save_mask, cal_metrics, batch_size = None, True, True, True, False, 1
    input_size = [512, 512] # None [1920, 1920] [1792, 1792] [1664, 1664] [1440, 1440] [768, 768]
    patch_size, patch_stride = None, None
    tta_scale = None  # [128, 192, 256]
    # test_img_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
    # test_gt_path = '/data1/zhengkengtao/docimg/docimg_split811/test_gt3/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split811/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/train_split811/test_gt/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/img/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train/mask/'
    # test_img_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_imgs/'
    # test_gt_path = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_gt/'
    # test_img_path = '/data1/zhengkengtao/docimg/tamper_mosaic/'
    # test_gt_path = '/data1/zhengkengtao/docimg/gt3/'
    # save_results_path = '/data1/zhengkengtao/exps/0730_denseFCN-nclass3_docimg_split811_png_noAug/densefcn_512x512/test_docimgsplit811test_bestckptepoch116/'
    # save_results_path = '/data1/zhengkengtao/exps/0730_denseFCN-nclass3_docimg_split811_png_noAug/densefcn_512x512/test_docimgsplit811test_bestckptepoch116_crop512stride256/'
    # checkpoint_dir = '/data1/zhengkengtao/exps/0730_denseFCN-nclass3_docimg_split811_png_noAug/densefcn_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/data1/zhengkengtao/exps/0730_denseFCN-nclass3_docimg_split811_png_noAug/densefcn_512x512/test_docimgsplit811test_robust/gaussnoise/mean0std5_crop512stride256/'
    # # ---------------
    # image_dataset_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig/'
    # gt_path = None
    # test_img_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_crop512stride256/'
    # test_gt_path = None
    # checkpoint_dir = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/' \
    #                     'test_docimgsplit811testorig/'
    # # # --------------
    # # ---------------
    # image_dataset_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic_jpg80/'
    # gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaicGT3/'
    # test_img_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaic_jpg80_crop512stride256/'
    # test_gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/testorig_mosaic/PSMosaicGT3_crop512stride256/'
    # checkpoint_dir = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/' \
    #                     'test_docimgsplit811testorig_mosaic_jpg80/'
    # # --------------
    # # ---------------
    # image_dataset_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper/'
    # gt_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/mask/'
    # test_img_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/tamper_crop512stride256/'
    # test_gt_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/mask_crop512stride256/'
    # checkpoint_dir = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/' \
    #                     'test_supat_all/'
    # # --------------
    # # ---------------
    # image_dataset_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/split19/test_imgs/'
    # gt_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/split19/test_gt/'
    # test_img_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/split19/test_imgs_crop512stride256/'
    # test_gt_path = '/pubdata/zhengkengtao/SUPATLANTIQUE/split19/test_gt_crop512stride256/'
    # checkpoint_dir = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/1426_supt_trainsplit19_ft_Aug/difnetv2_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/1426_supt_trainsplit19_ft_Aug/difnetv2_512x512/' \
    #                     'testbestepoch_supatsplit19/'
    # # # --------------
    # ---------------
    image_dataset_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_images/'
    gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_gt3/'
    test_img_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/test_images_crop512stride256/'
    test_gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/test_gt3_crop512stride256/'
    checkpoint_dir = '/pubdata/zhengkengtao/exps/1628_DIFNetV2-decoderSE_docimgsplit811_noAug/difnetv2_512x512/epoch60.pth'
    save_results_path = '/pubdata/zhengkengtao/exps/1628_DIFNetV2-decoderSE_docimgsplit811_noAug/difnetv2_512x512/' \
                        'testepoch60_docimgsplit811test/'
    # # --------------
    # # ---------------
    # image_dataset_path = '/pubdata/zhengkengtao/findit/T2tamper/'
    # gt_path = '/pubdata/zhengkengtao/findit/T2gt/'
    # test_img_path = '/pubdata/zhengkengtao/findit/T2tamper_crop512stride256/'
    # test_gt_path = '/pubdata/zhengkengtao/findit/T2gt_crop512stride256/'
    # checkpoint_dir = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/' \
    #                     'testbestepoch_finditT2tamperAll/'
    # # # --------------
    # # ---------------
    # image_dataset_path = '/pubdata/zhengkengtao/PS_arbitrary/tamper/'
    # gt_path = '/pubdata/zhengkengtao/PS_arbitrary/mask/'
    # test_img_path = '/pubdata/zhengkengtao/PS_arbitrary/tamper_crop512stride256/'
    # test_gt_path = '/pubdata/zhengkengtao/PS_arbitrary/mask_crop512stride256/'
    # checkpoint_dir = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/fold_0_best-ckpt.pth'
    # save_results_path = '/pubdata/zhengkengtao/exps/1315_DIFNetV2_docimgsplit811_crop512train_ConvNeXttiny-UperNet-gcAtt-noAug_add_encoderContraLoss_ppmf-temperature0.07_AugNoiseBlur/difnetv2_512x512/' \
    #                     'testbestepoch_PS_arbitrary/'
    # # # --------------
    map_path = save_results_path + 'map{}x{}/'.format(input_size[0], input_size[1])
    pred_path = save_results_path + 'pred{}x{}/'.format(input_size[0], input_size[1])
    # map_path = save_results_path + 'map/'
    # pred_path = save_results_path + 'pred/'
    # save_wordout_path = '/pubdata/zhengkengtao/docimg/docimg_split811/train0719_dif_noAug_wordcontour/difnet_1440x1440/test_bestckptepoch52/word_pred/'
    # pred_path = save_results_path + 'pred_noresize/'
    if (os.path.exists(map_path) == False): os.makedirs(map_path)
    if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
    logger = inial_logger(os.path.join(save_results_path, 'test.log'))
    filenames = os.listdir(test_img_path)
    filenames.sort()
    # filenames = filenames[75850:10*10000]
    mious, f1forgerys, iouforgerys, f1mosaics, ioumosaics = [], [], [], [], []
    t1 = time.time()
    # ------infer_batch start------
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_mask:{}, cal_metrics:{}, batch_size:{}'.format(n_class,mask_threshold, viz, save_mask, cal_metrics, batch_size))
    logger.info('test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames), input_size, patch_size, patch_stride, tta_scale))
    logger.info('checkpoint:{}'.format(checkpoint_dir))
    logger.info('test_img_path:{}'.format(test_img_path))
    logger.info('test_gt_path:{}'.format(test_gt_path))
    logger.info('save_test_results_path:{}'.format(save_results_path))
    logger.info('======================================================================================================')
    model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=3,  # 0728 更改为三通道输出
                     sobel=True,
                     constrain=True,
                     n_input=3,
                     )
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
    map_lst, pred_lst = [], []
    print('Total Test Batches: {}'.format(len(test_loader)))
    for batch_idx, batch_samples in enumerate(test_loader):
        img = batch_samples['image'].cuda()
        print('Testing batch {} ......'.format(batch_idx))
        with torch.no_grad():
            _, out = model(img)
        out = F.softmax(out, dim=1)  # [b,c,h,w]
        out = out.cpu().data.numpy()  # [b,c,h,w]
        pred = np.argmax(out, axis=1)  # [b,h,w]
        map = out[0]  # (c,h,w)
        map = np.array(map).transpose(1, 2, 0)  # (h,w,c)
        pred = pred[0]
        img_name = filenames[batch_idx]
        print(img_name)
        image = Image.open(test_img_path + img_name).convert('RGB')
        image = np.array(image)
        ht, wd, _ = image.shape
        if input_size is not None:
            map = cv2.resize(map, (wd, ht), interpolation=cv2.INTER_NEAREST)
            pred = cv2.resize(pred, (wd, ht), interpolation=cv2.INTER_NEAREST)
        pred = np.int64(pred)
        if save_map:
            map = Image.fromarray(np.uint8(map * 255))
            map = map.convert('RGB')
            map.save(map_path + img_name[:-4] + '.png')
        if save_mask:
            h, w = pred.shape[0], pred.shape[1]
            pred3 = np.ones([h, w, 3]) * 255
            pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
            pred3_1[pred == 1] = 0
            pred3_2[pred == 1] = 0
            pred3_0[pred == 2] = 0
            pred3_1[pred == 2] = 0
            pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
            pred3 = Image.fromarray(np.uint8(pred3))
            pred3 = pred3.convert('RGB')
            pred3.save(pred_path + img_name[:-4] + '.png')
    print('test time:', time.time()-t1)


    # ----------------merge----------------------
    from merge import get_row, get_col
    dst_dir = save_results_path
    block_read_path = dst_dir + 'pred512x512/'
    merge_write_path = dst_dir + 'predblockmerge/'
    mapblock_read_path = dst_dir + 'map512x512/'
    mapmerge_write_path = dst_dir + 'mapblockmerge/'

    if not os.path.exists(merge_write_path): os.makedirs(merge_write_path)
    if not os.path.exists(mapmerge_write_path): os.makedirs(mapmerge_write_path)
    image_name_list = os.listdir(image_dataset_path)
    block_name_list = os.listdir(block_read_path)
    mapblock_name_list = os.listdir(mapblock_read_path)

    block_size = 512
    step = 256
    image_name_list.sort()
    # image_name_list = image_name_list[0*100:1*100]
    print('merge num:{}'.format(len(image_name_list)))
    print(image_name_list)
    for image_name in image_name_list:
        print(image_name)
        image = cv2.imread(image_dataset_path + image_name, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[0], image.shape[1]
        # channel = image.shape[2]
        channel = 3
        block_name = [block for block in block_name_list if image_name.split(".", -1)[0] in block] # for DOC DID SUPAT
        mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] in mapblock] # for DOC DID SUPAT
        # block_name = [block for block in block_name_list if image_name.split(".", -1)[0] == block.split("_", -1)[0]] # for Ali
        # mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] == mapblock.split("_", -1)[0]] # for Ali
        block_name.sort()
        mapblock_name.sort()

        # ---合并概率图---
        mapmerge_block = np.zeros((height, width, channel))
        maptimes_save = np.zeros([height, width, channel])
        for mapblock in mapblock_name:
            mappredict_block = Image.open(mapblock_read_path + mapblock).convert('RGB')
            mappredict_block = np.array(mappredict_block, dtype=np.uint8)
            mappredict_block = mappredict_block
            the_row = get_row(mapblock)
            the_col = get_col(mapblock)
            right = block_size + int(the_col) * step
            left = right - block_size
            bottom = block_size + int(the_row) * step
            top = bottom - block_size
            if bottom <= height and right > width:
                y1, y2, x1, x2 = top, bottom, width - block_size, width
            elif bottom > height and right <= width:
                y1, y2, x1, x2 = height - block_size, height, left, right
            elif bottom > height and right > width:
                y1, y2, x1, x2 = height - block_size, height, width - block_size, width
            else:
                y1, y2, x1, x2 = top, bottom, left, right
            mapmerge_block[y1:y2, x1:x2] += mappredict_block
            maptimes_save[y1:y2, x1:x2] += 1
        mapmerge_block = mapmerge_block / maptimes_save
        mapmerge_block_ = Image.fromarray(np.uint8(mapmerge_block))
        mapmerge_block_ = mapmerge_block_.convert('RGB')
        mapmerge_block_.save(mapmerge_write_path + image_name[:-4] + '.png')

        # ---合并二值图---(不能合并每张pred再除以重复次数，因为会出现小数，无法分为三类)
        pred = np.argmax(mapmerge_block, axis=2)
        pred3 = np.ones([height, width, 3]) * 255
        pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
        pred3_1[pred == 1] = 0
        pred3_2[pred == 1] = 0
        pred3_0[pred == 2] = 0
        pred3_1[pred == 2] = 0
        pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
        pred3 = Image.fromarray(np.uint8(pred3))
        pred3 = pred3.convert('RGB')
        pred3.save(merge_write_path + image_name[:-4] + '.png')
    print("finish merge")


    # --------------cal metrics------------------------
    import csv
    save_metrics_path = save_results_path
    metrics_csv = save_metrics_path + "3kinds_metric.csv"
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(
        ['img', 'gt', 'f1-1', 'iou-1', 'mcc-1', 'auc-1', 'fpr-1', 'f1-2', 'iou-2', 'mcc-2', 'auc-2', 'fpr-2', 'miou',
         'mauc', 'pa', 'mpa'])
    f.close()
    filenames = os.listdir(merge_write_path)
    filenames.sort()
    mious, maucs, pas, mpas = [], [], [], []
    f1forgerys, iouforgerys, mccforgerys, aucforgerys, fprforgerys = [], [], [], [], []
    f1mosaics, ioumosaics, mccmosaics, auc_mosaics, fprmosaics = [], [], [], [], []
    for i in range(len(filenames)):
        name = filenames[i]

        # map = np.load(map_path + '{}.npy'.format(name[:-4]))
        map = Image.open(mapmerge_write_path + name)
        map = np.array(map) / 255
        # print(map[100, 100, 0] + map[100, 100, 1] + map[100, 100, 2])
        map_forgery = map[:, :, 1]
        map_mosaic = map[:, :, 2]

        pred = Image.open(merge_write_path + name).convert('L')
        pred = np.array(pred)
        pred[pred == 255] = 0
        pred[pred == 76] = 1
        pred[pred == 29] = 2
        if gt_path is not None:
            gt_name = name.replace('psc_', 'gt3_')
            gt_name = gt_name.replace('.jpg', '.png')
            gt_name = gt_name.replace('.tif', '.png')
            gt_name = gt_name.replace('_qf60', '')
            gt_name = gt_name.replace('_qf70', '')
            gt_name = gt_name.replace('_qf80', '')
            gt_name = gt_name.replace('_qf90', '')
            gt_name = gt_name.replace('ps', 'ms')
            gt = Image.open(gt_path + gt_name).convert('L')
            gt = np.array(gt)
            # docimg
            gt[gt == 255] = 0
            gt[gt == 76] = 1
            gt[gt == 29] = 2
            # # Alinew supatlantique certificate
            # gt[gt != 0] = 1
            # # findit-mosaic、supatlantique-mosaic
            # gt[gt !=0 ] = 2
        else:
            gt_name = ""
            h, w = pred.shape
            gt = np.zeros((h, w), dtype=np.uint8)
        metric = SegmentationMetric(3)  # 2表示有2个分类，有几个分类就填几
        hist = metric.addBatch(pred, gt)  # [h,w], [h,w]
        # IoU = metric.IntersectionOverUnion() # 每一个
        mIoU = metric.meanIntersectionOverUnion()
        pa = metric.pixelAccuracy()
        # cpa = metric.classPixelAccuracy() # 每一个
        mpa = metric.meanPixelAccuracy()  # CPA的平均
        # mauc = get_multiclass_mean_auc(gt, map) # 相加不为1算不了
        mauc = 0
        mious.append(mIoU)
        pas.append(pa)
        mpas.append(mpa)
        maucs.append(mauc)

        gtforgery, gtmosaic = gt.copy(), gt.copy()
        gtforgery[gtforgery == 2] = 0
        gtmosaic[gtmosaic == 1] = 0
        gtmosaic[gtmosaic == 2] = 1
        predforgery, predmosaic = pred.copy(), pred.copy()
        predforgery[predforgery == 2] = 0
        predmosaic[predmosaic == 1] = 0
        predmosaic[predmosaic == 2] = 1
        f11, iou1, mcc1, auc1, fpr1 = get_f1_iou_mcc_auc_fpr(gtforgery, predforgery, map_forgery)
        f12, iou2, mcc2, auc2, fpr2 = get_f1_iou_mcc_auc_fpr(gtmosaic, predmosaic, map_mosaic)

        print('{}, {}, f1forgery:{:.4f}, f1mosaic:{:.4f}'.format(i, name, f11, f12))
        print('======================================================================================================')
        f = open(metrics_csv, 'a+', newline='')
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [name, gt_name, f11, iou1, mcc1, auc1, fpr1, f12, iou2, mcc2, auc2, fpr2, mIoU, mauc, pa, mpa])
        f.close()

        f1forgerys.append(f11)
        iouforgerys.append(iou1)
        mccforgerys.append(mcc1)
        aucforgerys.append(auc1)
        fprforgerys.append(fpr1)
        f1mosaics.append(f12)
        ioumosaics.append(iou2)
        mccmosaics.append(mcc2)
        auc_mosaics.append(auc2)
        fprmosaics.append(fpr2)

    miou_mean, mauc_mean, pa_mean, mpa_mean = np.mean(mious), np.mean(maucs), np.mean(pas), np.mean(mpas)
    f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean = np.mean(f1forgerys), np.mean(
        iouforgerys), np.mean(mccforgerys), np.mean(aucforgerys), np.mean(fprforgerys)
    f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean = np.mean(f1mosaics), np.mean(
        ioumosaics), np.mean(mccmosaics), np.mean(auc_mosaics), np.mean(fprmosaics)
    print('======================================================================================================')
    print('test num:{}, f1foregry:{:.3f}, aucforgery:{:.3f}, fprforgery:{:.3f}, f1mosaic:{:.3f}, aucmosaic:{:.3f}, fprmosaic:{:.3f}'
        .format(len(filenames), f1forgery_mean, aucforgery_mean, fprforgery_mean, f1mosaic_mean, aucmosaic_mean, fprmosaic_mean))
    f = open(metrics_csv, 'a+', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Average', len(filenames),
                         f1forgery_mean, iouforgery_mean, mccforgery_mean, aucforgery_mean, fprforgery_mean,
                         f1mosaic_mean, ioumosaic_mean, mccmosaic_mean, aucmosaic_mean, fprmosaic_mean,
                         miou_mean, mauc_mean, pa_mean, mpa_mean])
    csv_writer.writerow(
        ['END', 'END', 'f1-1', 'iou-1', 'mcc-1', 'auc-1', 'fpr-1', 'f1-2', 'iou-2', 'mcc-2', 'auc-2', 'fpr-2', 'miou',
         'mauc', 'pa', 'mpa'])
    f.close()

    print('all time:', time.time() - t1)
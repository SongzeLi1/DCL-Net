import os
from utils.utils import inial_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from contdnet.contdnet import *
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
        image = np.array(image, np.uint8)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        return {
            'image': image
        }


if __name__=="__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    warnings.filterwarnings('ignore')
    n_class = 3
    mask_threshold, viz, save_map, save_mask, cal_metrics, batch_size = None, True, True, True, False, 1
    input_size = [512, 512]
    patch_size, patch_stride = None, None
    tta_scale = None
    # ---------------
    image_dataset_path = '/pubdata/1/docimg/docimg_split811/test_images/'
    gt_path = '/pubdata/1/docimg/docimg_split811/test_gt3/'
    test_img_path = '/pubdata/1/docimg/docimg_split811/crop512x512/test_images_crop512stride256/'
    test_gt_path = '/pubdata/1/docimg/docimg_split811/crop512x512/test_gt3_crop512stride256/'
    checkpoint_dir = '/pubdata/1/exps/bestval.pth'
    save_results_path = '/pubdata/1/exps/test_results/'
    # ---------------
    map_path = save_results_path + 'map{}x{}/'.format(input_size[0], input_size[1])
    pred_path = save_results_path + 'pred{}x{}/'.format(input_size[0], input_size[1])
    if (os.path.exists(map_path) == False): os.makedirs(map_path)
    if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
    logger = inial_logger(os.path.join(save_results_path, 'test.log'))
    filenames = os.listdir(test_img_path)
    filenames.sort()
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
    model = ConTDNet()
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
            out, _ = model(img) # DIFV2
        out = F.softmax(out, dim=1)
        out = out.cpu().data.numpy()
        pred = np.argmax(out, axis=1)
        map = out[0]
        map = np.array(map).transpose(1, 2, 0)
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
    print('merge num:{}'.format(len(image_name_list)))
    print(image_name_list)
    for image_name in image_name_list:
        print(image_name)
        image = cv2.imread(image_dataset_path + image_name, cv2.IMREAD_UNCHANGED)
        height, width = image.shape[0], image.shape[1]
        channel = 3
        block_name = [block for block in block_name_list if image_name.split(".", -1)[0] in block]
        mapblock_name = [mapblock for mapblock in mapblock_name_list if image_name.split(".", -1)[0] in mapblock]
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

        # ---合并二值图---
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
        map = Image.open(mapmerge_write_path + name)
        map = np.array(map) / 255
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
            gt = Image.open(gt_path + gt_name).convert('L')
            gt = np.array(gt)
            gt[gt == 255] = 0
            gt[gt == 76] = 1
            gt[gt == 29] = 2
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
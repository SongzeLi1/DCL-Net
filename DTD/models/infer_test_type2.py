import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
warnings.filterwarnings('ignore')
from utils.utils import inial_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from dtd import *
from utils.segmetric import *
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageFile
from utils.metric import *
import time
import logging
from dataset.DOCDataset_gt2s import DOCDataset

Image.MAX_IMAGE_PIXELS = 1000000000000000
ImageFile.LOAD_TRUNCATED_IMAGES = True

def infer_transform(input_size):
    normMean = [0.485, 0.456, 0.406]
    normStd = [0.229, 0.224, 0.225]
    if input_size is not None:
        return A.Compose([
            A.Resize(input_size[0], input_size[1]),  # width, height
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=normMean, std=normStd),
            ToTensorV2(),
        ])

if __name__=="__main__":
    warnings.filterwarnings('ignore')
    n_class = 2
    mask_threshold, viz, save_map, save_mask, batch_size = None, False, True, True, 160
    input_size = [512, 512] # None
    patch_size, patch_stride = None, None
    tta_scale = None  # [128, 192, 256]
    test_img_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/test_images_75_100/'
    test_gt_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/test_gt3_crop512stride256/'
    checkpoint_dir = "/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/dtd_512x512/fold_0_best-ckpt.pth"
    save_results_path = '/pubdata/lisongze/DCLNet/result/DTDNet_crop512x512/detection_type2/test_result_try/'
    map_path = save_results_path + 'map{}x{}/'.format(input_size[0], input_size[1])
    pred_path = save_results_path + 'pred{}x{}/'.format(input_size[0], input_size[1])
    if (os.path.exists(map_path) == False): os.makedirs(map_path)
    if (os.path.exists(pred_path) == False): os.makedirs(pred_path)
    logger = inial_logger(os.path.join(save_results_path, 'test_{}x{}.log'.format(input_size[0], input_size[1])))
    filenames = os.listdir(test_img_path)
    filenames.sort()
    # filenames = filenames[0*3232:1*3232]
    # mious, f1forgerys, iouforgerys, f1mosaics, ioumosaics = [], [], [], [], []
    mious, f1forgerys, iouforgerys = [], [], []
    t1 = time.time()
    # ------infer_batch start------
    logger.info('nclass:{}, mask_threshold:{}, viz:{}, save_mask:{}, batch_size:{}'.format(n_class,mask_threshold,viz,save_mask,batch_size))
    logger.info(
        'test_num:{}, input_size:{}, patch_size:{}, patch_stride:{}, tta_scale:{}'.format(len(filenames),input_size,patch_size,patch_stride,tta_scale))
    logger.info('checkpoint:{}'.format(checkpoint_dir))
    logger.info('test_img_path:{}'.format(test_img_path))
    logger.info('test_gt_path:{}'.format(test_gt_path))
    logger.info('save_test_results_path:{}'.format(save_results_path))
    logger.info(
        '======================================================================================================')
    model = seg_dtd('', 2).cuda()
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
    device = 'cuda'
    print('Total Test Batches: {}'.format(len(test_loader)))
    for batch_idx, batch_samples in enumerate(test_loader):
        img, dct, qs = batch_samples['image'], batch_samples['dct'], batch_samples['qs']
        img, dct, qs = Variable(img.to(device)), Variable(dct.to(device)), Variable(qs.unsqueeze(1).to(device))
        print('Testing batch {} ......'.format(batch_idx))
        with torch.no_grad():
            out = model(img, dct, qs)
        out = F.softmax(out, dim=1)  # [b,c,h,w]
        out = out.cpu().data.numpy()  # [b,c,h,w]
        pred = np.argmax(out, axis=1)  # [b,h,w]
        for i in range(pred.shape[0]):
            map_lst.append(out[i])  # out[i]:(c,h,w) map_lst:(t,c,h,w)
            pred_lst.append(pred[i])  # preds[i]:(h,w) pre_lst:(t,h,w)
    map_lst = np.array(map_lst).transpose(0, 2, 3, 1)
    pred_lst = np.array(pred_lst)
    np.save(save_results_path + 'map.npy', map_lst)
    np.save(save_results_path + 'pred.npy', pred_lst)
    # ------infer_batch end------
    map_lst = np.load(save_results_path + 'map.npy')
    pred_lst = np.load(save_results_path + 'pred.npy')
    for i in range(len(filenames)):
        map = map_lst[i]
        pred = pred_lst[i]
        img_name = filenames[i]
        print(img_name)
        image = Image.open(test_img_path + img_name).convert('RGB')
        image = np.array(image)
        ht, wd, _ = image.shape
        # if input_size is not None:
        #     map = cv2.resize(map, (wd, ht), interpolation=cv2.INTER_NEAREST)
        #     pred = cv2.resize(pred, (wd, ht), interpolation=cv2.INTER_NEAREST)
        pred = np.int64(pred)
        if save_map:
            map = Image.fromarray(np.uint8(map * 255))
            map = map.convert('RGB')
            # print(img_name[:-4]+'.png')
            map.save(map_path + img_name[:-4] + '.png')
        if save_mask:
            h, w = pred.shape[0], pred.shape[1]
            pred3 = np.zeros([h, w, 3])
            pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
            pred3_1[pred == 1] = 255  # all 1
            pred3_2[pred == 1] = 255
            pred3_0[pred == 1] = 255
            pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
            pred3 = Image.fromarray(np.uint8(pred3))
            pred3 = pred3.convert('RGB')
            pred3.save(pred_path + img_name[:-4] + '.png')
    print('test time:', time.time() - t1)


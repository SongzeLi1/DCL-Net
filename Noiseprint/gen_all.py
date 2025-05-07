import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import io
from time import time
from noiseprint.noiseprint_blind import noiseprint_blind_file
from noiseprint.utility.utilityRead import imread2f, computeMCC
from noiseprint.noiseprint_blind import genMappFloat
from PIL import Image


erodeKernSize  = 15
dilateKernSize = 11

mask_threshold = 0.5
# img_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
img_path = '/data1/zhengkengtao/docimg/tamper_mosaic/'
noise_path = '/data1/zhengkengtao/exps/0717_Noiseprint_docimg_all_test_png/mat/'
save_dir = '/data1/zhengkengtao/exps/0717_Noiseprint_docimg_all_test_png/'
save_txt = save_dir + 'test_error_img.txt'
save_pmap_path = save_dir + 'out_pmap/'
save_mask_path = save_dir + 'out_mask_thres{}/'.format(mask_threshold)
if not os.path.exists(save_dir): os.makedirs(save_dir)
if not os.path.exists(save_pmap_path): os.makedirs(save_pmap_path)
if not os.path.exists(save_mask_path): os.makedirs(save_mask_path)

imgfilename = r"C:\Users\CT\Desktop\yu\Noiseprint\img\ps_probe_chentong_android_honor_30_zhifubao_5_spl_donor_zhengkengtao_harmonyos_huawei_mate30_zhifubao_3.png"
outfilename = r'C:\Users\CT\Desktop\yu\Noiseprint\mat\ps_probe_chentong_android_honor_30_zhifubao_5_spl_donor_zhengkengtao_harmonyos_huawei_mate30_zhifubao_3.mat'

with open(imgfilename, 'rb') as f:
    stream = io.BytesIO(f.read())

timestamp = time()
QF, mapp, valid, range0, range1, imgsize, other = noiseprint_blind_file(imgfilename)
timeApproach = time() - timestamp

if mapp is None:
    print('Image is too small or too uniform')

out_dict = dict()
out_dict['QF'] = QF
out_dict['map'] = mapp
out_dict['valid'] = valid
out_dict['range0'] = range0
out_dict['range1'] = range1
out_dict['imgsize'] = imgsize
out_dict['other'] = other
out_dict['time'] = timeApproach


if __name__ == '__main__':
    ferror = open(save_txt, 'w+')
    for root, dirs, files in os.walk(img_path):
        files.sort()
        for name in tqdm(files):
            min_loss = 1000.0
            tamper = os.path.join(root, name)
            noise = noise_path + name.replace('png', 'mat')

            imgfilename = tamper
            outfilename = noise

            with open(imgfilename, 'rb') as f:
                stream = io.BytesIO(f.read())

            print(name)

            timestamp = time()
            QF, mapp, valid, range0, range1, imgsize, other = noiseprint_blind_file(imgfilename)
            timeApproach = time() - timestamp

            img, mode = imread2f(imgfilename, channel=3)
            mapp = genMappFloat(mapp, valid, range0, range1, imgsize)

            vmax = np.max(mapp)
            vmin = np.min(mapp)
            map_ct = (mapp - vmin) / (vmax - vmin)
            map_threshold = map_ct > mask_threshold

            pmap = Image.fromarray(np.uint8(map_ct * 255))
            pmap = pmap.convert('L')
            pmap.save(save_pmap_path + name[:-4] + '.png')
            mask = Image.fromarray(np.uint8(map_threshold * 255))
            mask = mask.convert('L')
            mask.save(save_mask_path + name[:-4] + '.png')

            # try:
            #     print(name)
            #     timestamp = time()
            #     QF, mapp, valid, range0, range1, imgsize, other = noiseprint_blind_file(imgfilename)
            #     timeApproach = time() - timestamp
            #
            #     img, mode = imread2f(imgfilename, channel=3)
            #     mapp = genMappFloat(mapp, valid, range0, range1, imgsize)
            #
            #     vmax = np.max(mapp)
            #     vmin = np.min(mapp)
            #     map_ct = (mapp - vmin) / (vmax - vmin)
            #     map_threshold = map_ct > mask_threshold
            #
            #     pmap = Image.fromarray(np.uint8(map_ct * 255))
            #     pmap = pmap.convert('L')
            #     pmap.save(save_pmap_path + name[:-4] + '.png')
            #     mask = Image.fromarray(np.uint8(map_threshold * 255))
            #     mask = mask.convert('L')
            #     mask.save(save_mask_path + name[:-4] + '.png')
            #
            # except:
            #     print('************************************************')
            #     print(name + ' Test Error !!!')
            #     print(name, file=ferror)
            #     img = Image.open(tamper).convert('RGB')
            #     img = np.array(img, dtype=np.uint8)
            #     [h, w, c] = img.shape
            #     pmap = np.zeros((h, w))
            #     pmap = Image.fromarray(np.uint8(pmap))
            #     pmap = pmap.convert('L')
            #     pmap.save(save_pmap_path + name[:-4] + '.png')
            #     mask = np.zeros((h, w))
            #     mask = Image.fromarray(np.uint8(mask))
            #     mask = mask.convert('L')
            #     mask.save(save_mask_path + name[:-4] + '.png')
            #     continue
    ferror.close()




import os
from skimage import io
import numpy as np
import warnings
from tqdm import tqdm
from PIL import Image, ImageFile
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


tamper_path = '/pubdata/1/docimg/tamper_mosaic/'
mask_path = '/pubdata/1/docimg/gt3/'
print(len(os.listdir(mask_path)))
size, stride = 512, 256 # 裁剪的图像大小, 步长
tamper_crop_path = '/pubdata/1/docimg/tamper_mosaic_crop{}stride{}/'.format(size, stride)
mask_crop_path = '/pubdata/1/docimg/gt3_crop{}stride{}/'.format(size, stride)
if not os.path.exists(tamper_crop_path): os.makedirs(tamper_crop_path)
if not os.path.exists(mask_crop_path): os.makedirs(mask_crop_path)
# tamper_crop_names = os.listdir(tamper_crop_path)
# print(len(tamper_crop_names))
imgs_name = os.listdir(tamper_path)
imgs_name.sort()
imgs_name = imgs_name[0*50: 1*50]
file_path = '/pubdata/1/docimg/crop{}stride{}_log.txt'.format(size, stride)
with open(file_path, 'r') as f:
    for line in f:
        cropped_img = line.strip('\n')
        imgs_name.remove(cropped_img)
with open(file_path, 'a') as f:
    for img_name in tqdm(imgs_name):
        print(img_name)
        img = Image.open(tamper_path + img_name).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        mask_name = img_name.replace('psc', 'gt3', 1)
        mask = Image.open(mask_path + mask_name).convert('RGB')
        mask = np.array(mask, dtype=np.uint8)
        h, w = img.shape[0], img.shape[1]
        h_num = int((h - size) / stride + 1)
        w_num = int((w - size) / stride + 1)
        h_lable = False
        w_lable = False
        h_max = (h_num - 1) * stride + size
        w_max = (w_num - 1) * stride + size
        assert h_max <= h
        assert w_max <= w
        if h_max < h:
            h_lable = True
        if w_max < w:
            w_lable = True
        num = 0
        for i in range(h_num + int(h_lable)):
            for j in range(w_num + int(w_lable)):
                num = num + 1
                h1 = i * stride
                h2 = i * stride + size
                w1 = j * stride
                w2 = j * stride + size
                if i == h_num:
                    h1 = h - size
                    h2 = h
                if j == w_num:
                    w1 = w - size
                    w2 = w
                patch = img[h1:h2, w1:w2, :]
                mask_patch = mask[h1:h2, w1:w2, :]
                io.imsave(tamper_crop_path + img_name[:-4] + '_%d_%d.png' % (i, j), patch)
                io.imsave(mask_crop_path + mask_name[:-4] + '_%d_%d.png' % (i, j), mask_patch)
        f.write(img_name + '\n')










import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import csv
from tqdm import tqdm
from time import sleep
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn import metrics
import cv2
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import *
import io
from noiseprint.noiseprint_blind import noiseprint_blind_file
from time import time
from noiseprint.utility.utilityRead import imread2f, computeMCC
from noiseprint.noiseprint_blind import genMappFloat
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from sklearn import metrics
erodeKernSize  = 15
dilateKernSize = 11


img_path = '/data1/zhengkengtao/docimg/docimg_split811/test_images/'
noise_path = '/data1/zhengkengtao/exps/0717_Noiseprint_docimg_split811_test_png/mat/'
save_mask_path = '/data1/zhengkengtao/exps/0717_Noiseprint_docimg_split811_test_png/out_mask/'
if not os.path.exists(save_mask_path): os.makedirs(save_mask_path)


if __name__ == '__main__':
    for root, dirs, files in os.walk(img_path):
        files.sort()
        for name in tqdm(files):
            if name not in os.listdir(save_mask_path):
                print(name)

                min_loss = 1000.0

                tamper = os.path.join(root, name)
                mask = tamper.replace('images','gt2s')

                noise = noise_path + name.replace('png', 'mat')

                imgfilename = tamper
                outfilename = noise
                reffilename = mask

                with open(imgfilename, 'rb') as f:
                    stream = io.BytesIO(f.read())

                print(tamper, noise)

                timestamp = time()
                QF, mapp, valid, range0, range1, imgsize, other = noiseprint_blind_file(imgfilename)
                timeApproach = time() - timestamp

                img, mode = imread2f(imgfilename, channel=3)
                mapp = genMappFloat(mapp, valid, range0, range1, imgsize)

                vmax = np.max(mapp)
                vmin = np.min(mapp)
                map_ct = (mapp - vmin) / (vmax - vmin)
                map_threshold = (map_ct > 0.5).astype(np.uint8)

                plt.figure()
                plt.imshow(map_threshold, cmap='gray')
                plt.show()

                smapp = mapp
                vmax = np.max(mapp)
                vmin = np.min(mapp)
                map_ct = (mapp - vmin) / (vmax - vmin)
                map_threshold = (map_ct > 0.5).astype(np.uint8)*255
                cv2.imwrite(save_mask_path + name, map_threshold)









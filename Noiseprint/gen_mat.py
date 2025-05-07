import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import csv
from tqdm import tqdm
from time import sleep
import random
from time import time
# from noiseprint.noiseprint import genNoiseprint
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv

# save_path = r'C:\Users\CT\Desktop\yu\Noiseprint\mat'
save_path = '/data1/zhengkengtao/exps/0717_Noiseprint_docimg_all_test_png/mat/'
if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    for root, dirs, files in os.walk('/data1/zhengkengtao/docimg/tamper_mosaic/'):
        for name in tqdm(files):
            tamper = os.path.join(root, name)
            name_mat = name.replace('png','mat')
            print(name_mat)
            # name_mat = name.replace('jpg', 'mat')

            imgfilename = tamper
            outfilename = os.path.join(save_path, name_mat)
            if os.path.exists(outfilename):
                continue

            timestamp = time()
            img, mode = imread2f(imgfilename, channel=1)

            try:
                # QF = jpeg_qtableinv(strimgfilenameeam)
                QF = jpeg_qtableinv(str(imgfilename))
            except:
                QF = 200
            res = genNoiseprint(img, QF)
            timeApproach = time() - timestamp

            out_dict = dict()
            out_dict['noiseprint'] = res
            out_dict['QF'] = QF
            out_dict['time'] = timeApproach

            if outfilename[-4:] == '.mat':
                import scipy.io as sio

                sio.savemat(outfilename, out_dict)
            else:
                import numpy as np

                np.savez(outfilename, **out_dict)
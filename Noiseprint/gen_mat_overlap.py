import os
import csv
from tqdm import tqdm
from time import sleep
import random
from time import time
# from noiseprint.noiseprint import genNoiseprint
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
import numpy as np
save_path = r'D:\datasets\NIST2016\mat_1'
if __name__ == '__main__':
    for root, dirs, files in os.walk(r"D:\datasets\NIST2016\tamper\manipulation_copy"):
        for name in tqdm(files):
            tamper = os.path.join(root, name)
            name_mat = name.replace('jpg','mat')
            print(name_mat)
            # name_mat = name.replace('jpg', 'mat')

            imgfilename = tamper
            outfilename = os.path.join(save_path, name_mat)
            if os.path.exists(outfilename):
                continue

            timestamp = time()
            img, mode = imread2f(imgfilename, channel=1)
            [x, y] = img.shape
            k = img[:, 0:50]
            img = np.concatenate((img, k), axis=1)

            try:
                # QF = jpeg_qtableinv(strimgfilenameeam)
                QF = jpeg_qtableinv(str(imgfilename))
            except:
                QF = 200
            res = genNoiseprint(img, QF)
            timeApproach = time() - timestamp
            res = res[:, 0:y]



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
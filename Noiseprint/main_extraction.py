# This is the code to extract Noiseprint
#    python main_extraction.py input.png noiseprint.mat
#    python main_showout.py input.png noiseprint.mat
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

from sys import argv
from time import time
# from noiseprint.noiseprint import genNoiseprint
from noiseprint.noiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv
import cv2

import matplotlib.pyplot as plt
import numpy as np

# imgfilename = argv[1]
#
# outfilename = argv[2]

# imgfilename = r'C:\Users\CT\Desktop\test1\tamper.png'
# outfilename = r'C:\Users\CT\Desktop\test1\tamper.mat'

imgfilename = r"C:\Users\CT\Desktop\yu\Noiseprint\img\ps_probe_chentong_android_honor_30_zhifubao_5_spl_donor_zhengkengtao_harmonyos_huawei_mate30_zhifubao_3.png"
outfilename = r'C:\Users\CT\Desktop\yu\Noiseprint\mat\ps_probe_chentong_android_honor_30_zhifubao_5_spl_donor_zhengkengtao_harmonyos_huawei_mate30_zhifubao_3.mat'


timestamp = time()
img, mode = imread2f(imgfilename, channel=1)

try:
    # QF = jpeg_qtableinv(strimgfilenameeam)
    QF = jpeg_qtableinv(str(imgfilename))
except:
    QF = 200
res = genNoiseprint(img,QF)
timeApproach = time() - timestamp



##############
[m,n] = res.shape
for i in range(0,m):
   for j in range(0,n):
       if(abs(res[i,j])>3):
           res[i,j]=0
################
print(res.max())
print(res.min())
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

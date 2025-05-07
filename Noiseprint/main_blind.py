# This code is the main of the noiseprint_blind
#    python main_blind.py input.png output.mat
#    python main_showout.py input.png output.mat
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
import scipy.io as sio
from time import time
import io
from noiseprint.noiseprint_blind import noiseprint_blind_file

# imgfilename = argv[1]
# outfilename = argv[2]
# imgfilename = r"D:\datasets\NIST2016\tamper\manipulation_copy\PSNC2016_1332_NC2016_0217.jpg"
# outfilename = r'C:\Users\CT\Desktop\noise_test\Sp_D_NRN_A_cha0003_pla0043_0538.mat'

imgfilename = r"C:\Users\CT\Desktop\yu\Noiseprint\img\ps_probe_chentong_android_honor_30_zhifubao_5_spl_donor_zhengkengtao_harmonyos_huawei_mate30_zhifubao_3.png"
outfilename = r'C:\Users\CT\Desktop\yu\Noiseprint\mat\ps_probe_chentong_android_honor_30_zhifubao_5_spl_donor_zhengkengtao_harmonyos_huawei_mate30_zhifubao_3.mat'

with open(imgfilename,'rb') as f:
    stream = io.BytesIO(f.read())
    
timestamp = time()
QF, mapp, valid, range0, range1, imgsize, other = noiseprint_blind_file(imgfilename)
timeApproach = time() - timestamp

if mapp is None:
    print('Image is too small or too uniform')

out_dict = dict()
out_dict['QF'     ] = QF
out_dict['map'    ] = mapp
out_dict['valid'  ] = valid
out_dict['range0' ] = range0
out_dict['range1' ] = range1
out_dict['imgsize'] = imgsize
out_dict['other'  ] = other
out_dict['time'   ] = timeApproach

if outfilename[-4:] == '.mat':
    import scipy.io as sio
    sio.savemat(outfilename, out_dict)
else:
    import numpy as np
    np.savez(outfilename, **out_dict)


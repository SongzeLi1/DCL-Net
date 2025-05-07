from sys import argv
from time import time
from noiseprint.mynoiseprint import genNoiseprint
from noiseprint.utility.utilityRead import imread2f
from noiseprint.utility.utilityRead import jpeg_qtableinv

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import cv2
# if __name__ == '__main__':
#
#     img = np.random.rand(3000,3000)
#
#     QF = 75
#     res = genNoiseprint(img, QF)
#     print(res.shape)
# dat = sio.loadmat(r"C:\Users\CT\Desktop\Text test\add\add\patchs\PSNC2016_0906_NC2016_1287.mat")['noiseprint']



# dat = sio.loadmat(r"C:\Users\CT\Desktop\Text test\add\add\iphone6_01_add.mat")['noiseprint']
# plt.figure()
# vmax = np.max(dat)
# vmin = np.min(dat)
# # index =map.argmax()
# dat = (dat-vmin)/(vmax-vmin)
# plt.imshow(dat, cmap='jet')
# plt.show()


# imgfilename =  r"C:\Users\CT\Desktop\testct\2.jpg"
# save =   r"C:\Users\CT\Desktop\testct\2.png"
#
# img_cv = cv2.imread(imgfilename)
# [m,n,c] = img_cv.shape
# mask = np.zeros((m,n))
# cv2.imwrite( save, mask)


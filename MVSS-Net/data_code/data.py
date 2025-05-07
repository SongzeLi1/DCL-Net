import os
import glob
import cv2
import numpy as np
# img_path = '/home/weishujin/Codes/pytorch/MVSS-Net-master/save_out/tianchi_test/mvssnet_casia/pred'
# output = '/home/weishujin/Codes/pytorch/MVSS-Net-master/save_out/tianchi_test/mvssnet_casia/pred_0.5'
img_path = '/home/weishujin/Codes/pytorch/MVSS-Net-master/save_out/tianchi_test/mvssnet_defacto/pred'
output = '/home/weishujin/Codes/pytorch/MVSS-Net-master/save_out/tianchi_test/mvssnet_defacto/pred_0.5'
imgs = glob.glob(img_path + '/*')
i = 0
for img in imgs:
    print(i)
    file = img.split('/')[-1]
    image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    print(np.max(image))
    image[image >= 127] = 255
    image[image < 127] = 0
    print(np.max(image))
    cv2.imwrite(os.path.join(output,file), image)
    i += 1
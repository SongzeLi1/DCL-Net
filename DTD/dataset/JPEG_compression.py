import os
import random

import cv2



path = "/pubdata/zhengkengtao/findit/findit_4types/split_100_20_600/crop512_stride256/test_imgs_crop512stride256/"
dst_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/findit/crop512_stride256/test_imgs_crop512stride256/'
if not os.path.exists(dst_path): os.makedirs(dst_path)
img_names = os.listdir(path)
for img_name in img_names:
    print(img_name)
    # quality = random.randint(75, 100)
    quality = 100
    img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
    # img_name = img_name.replace('psc_', 'psc_')
    cv2.imwrite(dst_path + img_name[:-4] + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])

import os
import random

import cv2



path = '/pubdata/zhengkengtao/docimg/docimg_split811/test_images_robust/resize/rate0.5/'
dst_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/test_robust/resize/rate0.5/'
if not os.path.exists(dst_path): os.makedirs(dst_path)
img_names = os.listdir(path)
for img_name in img_names:
    print(img_name)
    # quality = random.randint(75, 100)
    quality = 100
    img = cv2.imread(path + img_name, cv2.IMREAD_COLOR)
    # img_name = img_name.replace('psc_', 'psc_')
    cv2.imwrite(dst_path + img_name[:-4] + '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])

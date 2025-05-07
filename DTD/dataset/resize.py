import os
from PIL import Image

source_imgs_dir = '/pubdata/lisongze/docimg/exam/docimg2jpeg/Ali_new/resize_512x512/split_1000_200_2800/train_imgs/'
target_imgs_dir= "/pubdata/lisongze/docimg/exam/docimg2jpeg/test_robust/resize_gt3/rate0.5_resize256x256/"
if not os.path.exists(target_imgs_dir): os.makedirs(target_imgs_dir)
for file in os.listdir(source_imgs_dir):
    print(file)
    im = Image.open(source_imgs_dir + file)
    out = im.resize((256, 256), Image.ANTIALIAS)
    out.save(target_imgs_dir + file)

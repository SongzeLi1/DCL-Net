from PIL import Image
import os
import numpy as np


mask_threshold = 0.5
pmap_path = '/data1/zhengkengtao/exps/0718_handcraft_test_results/ADQ1/docimg_split811_test/Output_map/'
save_mask_path = '/data1/zhengkengtao/exps/0718_handcraft_test_results/ADQ1/docimg_split811_test/threshold_{}/'.format(mask_threshold)
if not os.path.exists(save_mask_path): os.makedirs(save_mask_path)
filename = os.listdir(pmap_path)
filename.sort()
N = len(filename)
for pmap_name in filename:
    print(pmap_name)
    pmap = Image.open(pmap_path + pmap_name).convert('L')
    pmap = np.array(pmap, dtype=np.float32) / 255
    mask = pmap > mask_threshold
    mask = Image.fromarray(np.uint8(mask*255))
    mask = mask.convert('L')
    mask.save(save_mask_path + pmap_name)
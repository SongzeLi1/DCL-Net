import warnings
from PIL import ImageFile
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import cv2
def get_row(block_name):
    the_row = block_name.split("_", -1)[-2]
    return the_row
def get_col(block_name):
    ending = block_name.split("_", -1)[-1]
    the_col = ending.split(".", -1)[0]
    return the_col
def patch_concat(img_path, patch_list, m1_all, m2_all, n1_all, n2_all):
    img_cv = cv2.imread(img_path)
    [M, N, C] = img_cv.shape
    img_save = np.zeros([M, N, C])
    time_save = np.zeros([M, N])
    len = m1_all.shape[0]
    for i in range(len):
        patch = patch_list[i]
        m1 = m1_all[i]
        m2 = m2_all[i]
        n1 = n1_all[i]
        n2 = n2_all[i]
        img_save[m1:m2, n1:n2] = img_save[m1:m2, n1:n2] + patch
        time_save[m1:m2, n1:n2] = time_save[m1:m2, n1:n2] + 1
    img = np.divide(img_save, time_save)
    return img


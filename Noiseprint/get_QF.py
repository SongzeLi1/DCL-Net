import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from noiseprint.utility.utilityRead import jpeg_qtableinv
import csv

# save_path = ''
# if not os.path.exists(save_path): os.makedirs(save_path)

if __name__ == '__main__':
    dir = '/data1/zhengkengtao/docimg/orig/'
    names = os.listdir(dir)
    names.sort()
    for name in names:
        imgfilename = dir + name
        try:
            QF = jpeg_qtableinv(str(imgfilename))
        except:
            QF = 200 # png图像无法调上面API
        print('name:{}, QF:{}'.format(name, QF))
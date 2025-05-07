import csv
import os
path = r"D:/文献阅读/文档类图像检测/工程/数据集/dpreview_dataset/data/1000/"
id = 6651739972
print(path+str(id)+".jpg")
print(os.path.exists(path+str(id)+".jpg"))
import os
path = r"D:\文献阅读\文档类图像检测\工程\数据集\dpreview_dataset\data\21000"
# singleimg = '7334253617'
for root,dirs,files in os.walk(path):
        for file in files:
            filename = os.path.join(root,file)
            if ".jpg.xltd" in filename or "(2)" in filename:
            # if singleimg in filename:
                print(filename)
                os.remove(filename)

import glob
import os
txt_path = '/home/weishujin/Codes/pytorch/MVSS-Net-master/data/tianchi_test.txt'
imgs = glob.glob('/pubdata/weishujin/TianChi/test/img/*.jpg')
print(len(imgs))
with open(txt_path,'w') as f:
    for img in imgs:
        # f.write(img + ' ' + img.replace('\img','\mask',1).replace('.jpg','.png',1)+ ' ' + '1\n')
        f.write(img + ' ' + 'None' + ' ' + '0\n')
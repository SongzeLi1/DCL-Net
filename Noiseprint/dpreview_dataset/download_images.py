# Script in Python 3.x to download the images from dpreview.com
db_root = r'D:\文献阅读\文档类图像检测\工程\数据集/dpreview.com/'        # directory where to save the downloaded images
file_images = './list_images.csv'  # list of images to download from dpreview.com

import requests
import pandas
import numpy as np
import time
import json
import os
import locale
from tqdm import tqdm
from urllib.request import urlretrieve
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
import socket
import os
import csv
# os.system(r'"D:\迅雷\Thunder\Program\ThunderStart.exe" {urls}'.format(urls=url))
def download(url, output_dir ):
    filename = os.path.basename(url).split('?')[0]
    output_file = os.path.join(output_dir, filename)
    print(url)
    with open(r".\images.csv", mode='w') as f:
        writer = csv.writer(f)
        writer.writerow([url])
    # os.system(r'"D:\迅雷\Thunder\Program\ThunderStart.exe" {urls}'.format(urls=url))
    # os.system("D:\迅雷\Thunder\Program\ThunderStart.exe" + url.format(urls=url))
    #######################################################################
    # try:
    #     urlretrieve(url, output_file)
    # except socket.timeout:
    #     count = 1
    #     while count <= 5:
    #         try:
    #             urlretrieve(url, output_file)
    #             break
    #         except socket.timeout:
    #             err_info = 'Reloading for %d time' % count if count == 1 else 'Reloading for %d times' % count
    #             print(err_info)
    #             count += 1
    #     if count > 5:
    #         print("downloading picture fialed!")
    ##########################################################################
    # trys = 0
    # while trys<6:
    #
    #     trys = trys+1
    #     try:
    #         # urlretrieve(url, output_file)
    #         print(url)
    #         os.system(r'"D:\迅雷\Thunder\Program\ThunderStart.exe" {urls}'.format(urls=url))
    #         return trys
    #     except:
    #         time.sleep(10)
    # return trys


list_images = pandas.read_csv(file_images, sep=';', dtype={'id': str,'id_model': str})
dat_models = dict()
print('Downloading images from dpreview.com ...')
with open(r".\images.csv", mode='w',newline='') as f:
    writer = csv.writer(f)
    for index, img in tqdm(list_images.iterrows(), total=len(list_images)):

        id_model = img['id_model']
        id_image = img['id']
        if id_model not in dat_models:
            output_dir = os.path.join(db_root, id_model)
            os.makedirs(output_dir, exist_ok=True)
            page0 = requests.get("https://www.dpreview.com/sample-galleries/data/get-gallery?galleryId=%s"%id_model)

            dat_models[id_model] = json.loads(page0.content)['images']

        try:
            info = [x for x in dat_models[id_model] if x['id']==id_image][0]

        except:
            print("下载失败文件id："+id_image)
            continue
        writer.writerow([info['url']])


print('Downloading is DONE.')
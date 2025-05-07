import os
from PIL import Image

folder_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/test_robust/resize/rate0.7_crop512stride256/' # "/pubdata/lisongze/docimg/exam/docimg2jpeg/test_images_75_100/"
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.npy']  # 可能的图片扩展名

num_images = 0  # 记录图片数量

# 遍历指定文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查文件是否是图片
    if any(file_name.lower().endswith(ext) for ext in image_extensions):
        num_images += 1
        # 打印图片文件名
        # print(file_name)

# 打印图片数量
print(f"Number of images: {num_images}")
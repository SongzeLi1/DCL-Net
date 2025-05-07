from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import random
import os
import cv2
import torch
import torchvision.transforms.functional as F


def time_calucate(star_time,end_time,epoch,total_epoch,step,total_step):
    seconds = end_time - star_time
    present_sec = (total_step - step) * seconds
    total_sec = (total_epoch-epoch) * total_step * seconds + present_sec
    present_min, present_sce = divmod(present_sec, 60)
    present_hour, present_min = divmod(present_min, 60)
    total_min, total_sec = divmod(total_sec, 60)
    total_hour, total_min = divmod(total_min, 60)
    total_day = total_hour//24
    total_hour = total_hour - (total_hour//24)*24
    return present_hour,present_min,total_day, total_hour,total_min


def data_path(args):
    root = os.path.join(args.data_root, args.compression,args.data, '$/png/*')
    train_root = root.replace('$', 'train', 1)
    valid_root = root.replace('$', 'val', 1)
    test_root  = root.replace('$', 'test', 1)
    return root, train_root, valid_root, test_root

def flow_output(fn):
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            data = np.fromfile(f, np.float32, count=2 * w * h)

            return np.reshape(data, (h, w, 2))

normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}

def img_to_tensor(im, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(im / (255.0 if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor

def imgs_loader(imgs_path,resize=None):
    imgs = []
    for img in imgs_path:
        img = Image.open(img)
        img = img.convert("RGB")
        img = np.array(img)
        if resize!=None:
            img = cv2.resize(img, resize)
        img = img/255.
        img = img.astype(np.float32)
        imgs.append(img)
    return np.array(imgs)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 椭圆结构

def labels_loader(labels_path,resize=None):
    label = Image.open(labels_path)
    label = label.convert("L")
    if resize != None:
        label = label.resize(resize)
    label = np.array(label)

    label = label[...,np.newaxis]
    label = label/255.
    label[label >= 0.5] = 1
    label[label < 0.5] = 0
    label = label.astype(np.int32)

    return label

def get_fre(img_path):
    w = 224
    h = 224
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # 提取频谱
    img = np.float32(img) / 255
    img_dct = cv2.dct(img)

    ##### 低频 #####
    img_dct1 = cv2.dct(img)
    low_area = np.array([[[0,0], [0,h//16], [w//16,0]]], dtype=np.int32)
    cv2.fillPoly(img_dct1, low_area, 0) # 绘制多边形并填充
    img_dct1 = img_dct - img_dct1

    ##### 中频 #####
    img_dct2 = cv2.dct(img)
    mid_area = np.array([[[0,h//16], [0,h//8], [w//8,0], [w//16,0]]], dtype=np.int32)
    cv2.fillPoly(img_dct2, mid_area, 0)
    img_dct2 =  img_dct - img_dct2

    ##### 高频 #####
    img_dct3 = cv2.dct(img)
    high_area = np.array([[[0,0], [0,h//8], [w//8,0]]])
    cv2.fillPoly(img_dct3, high_area, 0)

    img_idct1 = torch.Tensor(cv2.idct(img_dct1)).view(1, 224, 224)
    img_idct2 = torch.Tensor(cv2.idct(img_dct2)).view(1, 224, 224)
    img_idct3 = torch.Tensor(cv2.idct(img_dct3)).view(1, 224, 224)
    # cv2.imshow('low', img_idct1)

    # 按通道将三个频率分量的IDCT图像concatenate在一起 [3, 224, 224]
    concat = torch.cat([img_idct1, img_idct2, img_idct3], 0)
    return concat


def flows_loader(flows_path):
    flows = []
    for path in flows_path:
        flow = flow_output(path)
        flows.append(flow)
    flows = np.stack(flows, axis=0)
    return flows

def read_flow_image_window(path, pattern, window_size=3, stride=4):

    imgs = []
    k = window_size//2
    for i in range(window_size+stride-1):
        imgs.append([])
    img_names = glob.glob(os.path.join(path, pattern))
    img_names.sort()
    assert (len(img_names)>= stride), 'The num of images is not enough in {}.'.format(path)
    for i in range((len(img_names)-2*k)//stride):
        for j in range(len(imgs)):
            imgs[j] += [img_names[i*stride+j]]
    if (len(img_names)-2*k) % stride != 0:
        for j in range(len(imgs)):
            imgs[j] += [img_names[-1 * len(imgs)+j]]
    return imgs

def transpose(matrix):
    return zip(*matrix)

def read_dataset_flow_window(args, data_dir, mode,pattern ,print_func,shuffle_seed=None):

    imgs, labels = [],[[],[],[],[]]
    step = 4
    msk_replace = [['png', 'msk']]
    window_size = args.window_size
    for i in range(window_size+step-1):
        imgs.append([])
    image_names = glob.glob(os.path.join(data_dir, '*'))
    image_names.sort()
    if mode=='train':
        random.seed(shuffle_seed)
        random.shuffle(image_names)
    print_func('The num of ' + mode + ' files: {}'.format(len(image_names)))
    for i in range(len(image_names)):
        n = read_flow_image_window(image_names[i], pattern, window_size=window_size, stride=4)
        for j in range(len(imgs)):
            imgs[j] += n[j]
    instance_num = len(imgs[0])
    print_func('The num of ' + mode + ' slices:{} '.format(instance_num))
    for entry in msk_replace:
        for j in range(len(labels)):
            labels[j] = [name.replace(entry[0], entry[1], 1) for name in imgs[j+(window_size//2)]]

    if args.flow_generator == 'N':
        flows = []
        for i in range((window_size // 2) * 8):
            flows.append([])
        for i in range(step):
            if window_size == 5:
                flows[i] = [name.replace('png', 'forward', 1).replace('.png', '.flo', 1) for name in imgs[i+2]]
                flows[i+4] = [name.replace('png', 'backward', 1).replace('.png', '.flo', 1) for name in imgs[i+1]]
                flows[i+8] = [name.replace('png', 'forward_stride2', 1).replace('.png', '.flo', 1) for name in imgs[i+2]]
                flows[i+12] = [name.replace('png', 'backward_stride2', 1).replace('.png', '.flo', 1) for name in imgs[i]]
            elif window_size == 7:
                flows[i] = [name.replace('png', 'forward', 1).replace('.png', '.flo', 1) for name in imgs[i + 3]]
                flows[i+4] = [name.replace('png', 'backward', 1).replace('.png', '.flo', 1) for name in imgs[i + 2]]
                flows[i+8] = [name.replace('png', 'forward_stride2', 1).replace('.png', '.flo', 1) for name in imgs[i+3]]
                flows[i+12] = [name.replace('png', 'backward_stride2', 1).replace('.png', '.flo', 1) for name in imgs[i+1]]
                flows[i+16] = [name.replace('png', 'forward_stride3', 1).replace('.png', '.flo', 1) for name in imgs[i+3]]
                flows[i+20] = [name.replace('png', 'backward_stride3', 1).replace('.png', '.flo', 1) for name in imgs[i]]
        imgs = list(transpose(imgs))
        labels = list(transpose(labels))
        flows = list(transpose(flows))
        return imgs, labels, flows
    imgs = list(transpose(imgs))
    labels = list(transpose(labels))
    return imgs, labels



def read_dataset(data_dir, mode,pattern ,print_func,shuffle_seed=None):

    imgs = []
    for dir in data_dir:
        dir_imgs_ = glob.glob(os.path.join(dir, pattern))
        dir_imgs_.sort()
        if len(dir_imgs_)%3 != 0:
            dir_imgs_ = dir_imgs_[:(len(dir_imgs_)//3)*3] + dir_imgs_[-3:]
        dir_imgs = [dir_imgs_[i:i+3] for i in range(0,len(dir_imgs_),3)]
        imgs.extend(dir_imgs)
    if mode=='train':
        random.seed(shuffle_seed)
        random.shuffle(imgs)
    print_func('The num of ' + mode + ' clip: {}'.format(len(imgs)))

    labels = [imgs_[1].replace('png','msk',1) for imgs_ in imgs]

    return imgs, labels


class MyDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, imgs, labels, resize=None, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.imgs_loader = imgs_loader
        self.labels_loader = labels_loader
        self.fre_loader = get_fre
        self.resize = resize

    def __getitem__(self, index):
        imgs_path = self.imgs[index]  #['path1','path2','path3']
        labels_path = self.labels[index]  #'path'
        imgs_ = self.imgs_loader(imgs_path,self.resize)  # 3,channel,h,w
        labels_ = self.labels_loader(labels_path,self.resize)  # channel,h,w
        fres_ = self.fre_loader(imgs_path[1])
        return imgs_, labels_, fres_, labels_path

    def __len__(self):
        return len(self.imgs)


def image_inpaint(image,mask_file,name):
    "image shape:[h, w, c]"
    "mask_file 是所有mask文件夹路径"
    h, w, c = image.shape
    mask = glob.glob(os.path.join(mask_file,'*.png'))
    mask.sort()
    num = random.randint(0, len(mask))
    mask = mask[num]
    mask = cv2.imread(mask, 0)
    mask = cv2.resize(mask,(w,h))
    inpaint_img = cv2.inpaint(image, mask[...,np.newaxis],3 , flags=cv2.INPAINT_TELEA)
    cv2.imwrite('/pubdata/weishujin/Ali/inpaint_result/' + name + '.png',image)
    cv2.imwrite('/pubdata/weishujin/Ali/inpaint_result/' + name + '_inpaint.png', inpaint_img.astype(np.uint8))
    cv2.imwrite('/pubdata/weishujin/Ali/inpaint_result/' + name + '_mask.png', mask.astype(np.uint8))
    mask = mask/ 255
    return inpaint_img, mask.astype(np.uint8)

# if __name__ == '__main__':
#     image_paths = glob.glob('/pubdata/weishujin/Ali/image/*')
#     image_paths.sort()
#     for i in range(10):
#         image = image_paths[i]
#         name = image.split('/')[-1].split('.')[0]
#         mask_file = '/pubdata/weishujin/Ali/mask'
#         image = cv2.imread(image)
#         image_inpaint(image,mask_file,name)
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import random
import os
import cv2
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import Resize

# torchresize = Resize(labels_.shape[0] // 4)
torchresize = Resize(512 // 4)

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

def imgs_loader(imgs_path, resize=None):
    # img = Image.open(imgs_path)
    # img = img.convert("RGB")
    img = cv2.imread(imgs_path)
    if resize!=None:
        # img = img.resize(resize)
        img = cv2.resize(img, resize)
    img = img_to_tensor(img, normalize)

    # img = np.array(img)
    # img = img/255.
    # img = img.astype(np.float32)
    return img

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 椭圆结构

def labels_loader(labels_path,resize,use_sobel=None):
    label = Image.open(labels_path)
    label = label.convert("L")
    if resize != None:
        label = label.resize(resize)
        label = np.array(label)
        label_ = label.copy()
        label_[label_ == 255] = 0
        label_[label_ == 76] = 1
        label_[label_ == 29] = 2
        dst = cv2.resize(label_, resize)
        dst = np.uint8(dst)
        pred33 = np.ones([512, 512, 3]) * 255
        pred33_0, pred33_1, pred33_2 = pred33[:, :, 0], pred33[:, :, 1], pred33[:, :, 2]
        pred33_1[dst == 1] = 0
        pred33_2[dst == 1] = 0
        pred33_0[dst == 2] = 0
        pred33_1[dst == 2] = 0
        pred33[:, :, 0], pred33[:, :, 1], pred33[:, :, 2] = pred33_0, pred33_1, pred33_2
        label_p = pred33
        label_p = Image.fromarray(np.uint8(label_p))
        label = label_p.convert('L')

    label = np.array(label)
    h, w = label[0], label[1]
    if use_sobel != None:
        # dilation = cv2.dilate(label, kernel)
        # dilation = dilation[...,np.newaxis]
        # edg_label = dilation - label
        # edg_label[edg_label<0] = 0
        # edg_label = cv2.resize(edg_label, (label.shape[1]//4,label.shape[0]//4))
        if resize is not None:
            label_0 = label.copy()
            label_0[label_0 == 255] = 0
            label_0[label_0 == 76] = 1
            label_0[label_0 == 29] = 2
            dstimg = cv2.resize(label_0, resize)
            dstimg = np.uint8(dstimg)
            pred3 = np.ones([512, 512, 3]) * 255
            pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
            pred3_1[dstimg == 1] = 0
            pred3_2[dstimg == 1] = 0
            pred3_0[dstimg == 2] = 0
            pred3_1[dstimg == 2] = 0
            pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
            label_01 = pred3
            label_01 = Image.fromarray(np.uint8(label_01))
            label_01 = label_01.convert('L')
            label_01 = np.array(label_01)
            # print(label_01.min(), label_01.max())
        else:
            label_01 = label.copy() # copy不能少，不然label会被label_01更新

        label_01[label_01 == 255] = 0
        label_01[label_01 == 76] = 255
        label_01[label_01 == 29] = 255

        edg_label = np.zeros([label_01.shape[0],label_01.shape[1]],dtype=np.uint8)
        _, binary = cv2.threshold(label_01, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edg_label, contours, -1, (255, 255, 255), 3)
        edg_label = edg_label[...,np.newaxis]
        edg_label = edg_label / 255.
        edg_label = cv2.resize(edg_label, (label_01.shape[1] // 4, label_01.shape[0] // 4))
        edg_label[edg_label >= 0.5] = 1
        edg_label[edg_label < 0.5] = 0
        edg_label = edg_label.astype(np.int32)
    else:
        edg_label = 0
    label = label[..., np.newaxis]
    label[label == 255] = 0
    label[label == 76] = 1
    label[label == 29] = 2
    label = label.astype(np.int32)
    # cv2.imwrite('/home/weishujin/Codes/pytorch/MVSS-Net-master/save_out/test__/edge.png',(edg_label[...,0] * 255).astype(np.uint8))
    # cv2.imwrite('/home/weishujin/Codes/pytorch/MVSS-Net-master/save_out/test__/label.png', (label[...,0] * 255).astype(np.uint8))
    # print('^^^^####:', label.min(),label.max(),edg_label.min(),edg_label.max())
    return label, edg_label

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

def read_dataset(args, data_dir, mode,pattern ,print_func,shuffle_seed=None):
    imgs = glob.glob(os.path.join(data_dir[0], pattern))
    if len(data_dir) >= 2:
        for i in range(1, len(data_dir)):
            imgs += glob.glob(os.path.join(data_dir[i], pattern))
    imgs.sort()
    # imgs = imgs[:8]
    if mode=='train':
        random.seed(shuffle_seed)
        random.shuffle(imgs)
    print_func('The num of ' + mode + ' files: {}'.format(len(imgs)))
    if 'Inpainting_dataset' in data_dir[0]:
        labels = [img.replace('png', 'msk', 1) for img in imgs]
    else:
        if mode != 'visual':
            if pattern == '*.png':
                # # docimg png
                # labels = [img.replace('images', 'gt3', 1).replace('psc', 'gt3', 1) for img in imgs]
                # docimg printer png
                labels = [img.replace('imgs', 'gt3', 1).replace('psc', 'gt3', 1) for img in imgs]
                # # # findit4types
                # labels = [img.replace('_imgs', '_gt3', 1) for img in imgs]
                # labels = [img.replace('jpg', 'png', 1) for img in labels]
                # labels = [gt3name[:gt3name.index(gt3name.split('/')[-1])] + 'gt3_' + gt3name.split('/')[-1] for gt3name in labels]
            elif pattern == '*.jpg':
                # # Alinew2types
                labels = [img.replace('_imgs', '_gt3', 1) for img in imgs]
                labels = [img.replace('jpg', 'png', 1) for img in labels]
                labels = [gt3name[:gt3name.index(gt3name.split('/')[-1])] + 'gt3_' + gt3name.split('/')[-1] for
                          gt3name in labels]
            else:
                labels = [img.replace('imgs', 'gt', 1).replace('.jpg', '.png', 1) for img in imgs] # Alinew
        else:
            labels = [None]*len(imgs)

    return imgs, labels


class MyDataset(Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, imgs, labels, use_sobel, resize=None, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.imgs_loader = imgs_loader
        self.labels_loader = labels_loader
        self.resize = resize
        self.use_sobel = use_sobel

    def __getitem__(self, index):
        imgs_path = self.imgs[index]
        labels_path = self.labels[index]
        if self.transform is not None:
            image = Image.open(imgs_path).convert('RGB')
            image = np.array(image, np.uint8)
            if labels_path != None:
                label = Image.open(labels_path)
                label = label.convert("L")
                if self.resize != None:
                    label = label.resize(self.resize)
                label = np.array(label)
                if self.use_sobel != None:
                    if self.resize is not None:
                        label_0 = label.copy()
                        label_0[label_0 == 255] = 0
                        label_0[label_0 == 76] = 1
                        label_0[label_0 == 29] = 2
                        dstimg = cv2.resize(label_0, (512, 512))
                        dstimg = np.uint8(dstimg)
                        pred3 = np.ones([512, 512, 3]) * 255
                        pred3_0, pred3_1, pred3_2 = pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2]
                        pred3_1[dstimg == 1] = 0
                        pred3_2[dstimg == 1] = 0
                        pred3_0[dstimg == 2] = 0
                        pred3_1[dstimg == 2] = 0
                        pred3[:, :, 0], pred3[:, :, 1], pred3[:, :, 2] = pred3_0, pred3_1, pred3_2
                        label_01 = pred3
                        label_01 = Image.fromarray(np.uint8(label_01))
                        label_01 = label_01.convert('L')
                        label_01 = np.array(label_01)
                    else:
                        label_01 = label.copy()  # copy不能少，不然label会被label_01更新

                    label_01[label_01 == 255] = 0
                    label_01[label_01 == 76] = 255
                    label_01[label_01 == 29] = 255

                    edg_label = np.zeros([label_01.shape[0], label_01.shape[1]], dtype=np.uint8)
                    _, binary = cv2.threshold(label_01, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(edg_label, contours, -1, (255, 255, 255), 3)
                    edg_label = edg_label[..., np.newaxis]
                    edg_label = edg_label / 255.
                    # edg_label = cv2.resize(edg_label, (label_01.shape[1] // 4, label_01.shape[0] // 4))
                    edg_label[edg_label >= 0.5] = 1
                    edg_label[edg_label < 0.5] = 0
                    edg_labels_ = edg_label.astype(np.int32)
                else:
                    edg_labels_ = 0
                label = label[..., np.newaxis]
                label[label == 255] = 0
                label[label == 76] = 1
                label[label == 29] = 2
                labels_ = label.astype(np.int32)
            else:
                labels_, edg_labels_ = 0., 0.
            gts = np.zeros((labels_.shape[0], labels_.shape[1], 1, 2))
            gts[:,:,:,0] = labels_
            gts[:,:,:,1] = edg_labels_
            transformed = self.transform(image=image, mask=gts)
            imgs_ = transformed['image']
            gts = transformed['mask']
            labels_ = gts[:,:,:,0]
            edg_labels_ = gts[:,:,0,1]
            edg_labels_ = torch.unsqueeze(edg_labels_, dim=0)
            edg_labels_ = torch.unsqueeze(edg_labels_, dim=0)
            edg_labels_ = torchresize(edg_labels_)
            edg_labels_ = edg_labels_[0,0,:,:]
            edg_labels_[edg_labels_ >= 0.5] = 1
            edg_labels_[edg_labels_ < 0.5] = 0

            # print(labels_.min(), labels_.max())

        else:
            imgs_ = self.imgs_loader(imgs_path, self.resize)  # 3,channel,h,w
            # imgs_ = self.imgs_loader(imgs_path)
            if labels_path != None:
                labels_, edg_labels_ = self.labels_loader(labels_path,self.resize, self.use_sobel)  # channel,h,w
            else:
                labels_, edg_labels_ = 0., 0.
        return imgs_, labels_, edg_labels_, imgs_path
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
    inpaint_img = cv2.inpaint(image, mask[...,np.newaxis], 3 , flags=cv2.INPAINT_TELEA)
    cv2.imwrite('/pubdata/weishujin/Ali/inpaint_result/' + name + '.png',image)
    cv2.imwrite('/pubdata/weishujin/Ali/inpaint_result/' + name + '_inpaint.png', inpaint_img.astype(np.uint8))
    cv2.imwrite('/pubdata/weishujin/Ali/inpaint_result/' + name + '_mask.png', mask.astype(np.uint8))
    mask = mask / 255
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
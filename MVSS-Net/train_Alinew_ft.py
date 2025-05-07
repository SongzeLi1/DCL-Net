import os
import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import sys
import time
import numpy as np
import glob
import cv2


from torch.utils.data import DataLoader
# from model import VIDSAT, Random_mask_guide
# from RAFT_master.core.raft_modify import RAFT_modify
from utils import read_dataset
from shutil import rmtree
from tee_print import TeePrint
from utils import MyDataset, time_calucate, data_path
from loss_function import DiceLoss, iou_compute
from sklearn.metrics import f1_score
from models.mvssnet import get_mvss
from skimage import io

class Compute_metric(object):  # 创建Circle类
   def __init__(self): # 初始化一个属性r（不要忘记self参数，他是类下面所有方法必须的参数）
       super(Compute_metric, self).__init__()
       self.epoch_loss = []
       self.correct = []
       self.f1 = []
       self.iou = []

   def f1(self, labels, pred_map):
        labels = labels.reshape([-1]).astype('int')
        pred_map = pred_map.reshape([-1]).astype('int')
        self.f1.append(f1_score(labels, pred_map))
   def iou(self, labels, pred_map):
        self.iou.append(self.iou_compute(labels, pred_map))

   def iou_compute(self, y_true, y_pred):
       tp = np.sum(np.logical_and(y_true, y_pred)).astype(np.float64)
       fn = np.sum(np.logical_and(y_true, np.logical_not(y_pred))).astype(np.float64)
       fp = np.sum(np.logical_and(np.logical_not(y_true), y_pred)).astype(np.float64)
       iou = tp / (fp + tp + fn)
       return iou


def split(images):
    b, c, h, w = images.shape
    blocks = []
    h1 = np.ceil(h / 512.)
    w1 = np.ceil(w / 512.)


import numpy as np

def read_image_block_without_mask(images,IMG_SIZE,step):
    images = np.squeeze(images,axis = 0)
    image_size = images.shape
    row,col,ch = image_size[0],image_size[1],image_size[2]

    test_blocks = []
    for i in np.arange(0, row - IMG_SIZE, step, dtype=np.int):
        for j in np.arange(0, col - IMG_SIZE, step, dtype=np.int):
            test_blocks.append(images[i:np.int(i + IMG_SIZE), j:np.int(j + IMG_SIZE),:])
        test_blocks.append(images[i:np.int(i + IMG_SIZE), np.int(col - IMG_SIZE):col,:])
    for j in np.arange(0, int(col - IMG_SIZE), step, dtype=np.int):
        test_blocks.append(images[np.int(row - IMG_SIZE):row, j:np.int(j + IMG_SIZE),:])
    test_blocks.append(images[row - IMG_SIZE:row, col - IMG_SIZE:col,:])
    test_blocks = np.array(test_blocks).reshape(-1,IMG_SIZE,IMG_SIZE,3)

    return test_blocks,row,col,images

def consist_whole_mask(test_results,row,col,step,IMG_SIZE):
    mask = np.zeros((row,col))
    num_every_pixel_scan = np.zeros((row,col))
    count = 0
    for i in np.arange(0,row - IMG_SIZE,step,dtype = np.int):
        for j in np.arange(0,col-IMG_SIZE,step,dtype = np.int):
            mask[i:np.int(i+IMG_SIZE),j:np.int(j+IMG_SIZE)] += test_results[count]
            num_every_pixel_scan[i:np.int(i+IMG_SIZE),j:np.int(j+IMG_SIZE)] += 1
            count+=1
        mask[i:np.int(i+IMG_SIZE),np.int(col-IMG_SIZE):col] += test_results[count]
        num_every_pixel_scan[i:np.int(i + IMG_SIZE), np.int(col - IMG_SIZE):col] += 1
        count += 1
    for j in np.arange(0,int(col-IMG_SIZE),step,dtype=np.int):
        mask[np.int(row-IMG_SIZE):row,j:np.int(j+IMG_SIZE)] += test_results[count]
        num_every_pixel_scan[np.int(row-IMG_SIZE):row,j:np.int(j+IMG_SIZE)] +=1
        count+=1
    mask[row-IMG_SIZE:row,col-IMG_SIZE:col] += test_results[count]
    num_every_pixel_scan[row-IMG_SIZE:row,col-IMG_SIZE:col] += 1
    mask = mask/num_every_pixel_scan
    return mask

def visual(test_loader,mode):
    model.eval()
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            images, _ ,_ , names = data
            b, c, h, w = images.shape
            images = images.to(device)
            if mode == 'joint':
                assert b == 1, 'Plese set batch size 1'
                images = images.permute([0, 2, 3, 1])
                images = images.cpu().detach().numpy()
                images, row, col, img = read_image_block_without_mask(images, 512, 256)
                images = torch.Tensor(images).to(device)
                images = images.permute([0, 3, 1, 2])
            _, outputs = model(images, train_mode='test')
            outputs = torch.sigmoid(outputs).to(device)
            if mode == 'joint':
                pred = outputs.permute([0, 2, 3, 1])
                pred = pred.cpu().detach().numpy()
                pred = pred.squeeze(-1)
                pred = consist_whole_mask(pred, row, col, step=256, IMG_SIZE=512)
                pred = (torch.Tensor(pred)).to(device)
            else:
                pred = outputs.permute([0, 2, 3, 1])
            pred = pred.cpu().detach().numpy()
            pred = pred.squeeze(-1)
            if mode == 'joint':
                pred_map = consist_whole_mask(pred,row,col,step=256,IMG_SIZE=512)
            else:
                pred_map = pred
            pred_map[pred_map >= 0.5] = 1
            pred_map[pred_map < 0.5] = 0
            out = r'/pubdata/weishujin/Deep_inpainting_localization/visual/MVSS'
            file, imgname = names[0].split('/')[-2], names[0].split('/')[-1]
            if not os.path.exists(os.path.join(out, file, 'pred')):os.makedirs(os.path.join(out, file, 'pred'))
            if not os.path.exists(os.path.join(out, file, 'pred_mask')):os.makedirs(os.path.join(out, file, 'pred_mask'))
            io.imsave(os.path.join(out, file, 'pred', imgname.replace('.png', '_pred.png')),  np.uint8(pred * 255))
            io.imsave(os.path.join(out, file, 'pred_mask', imgname.replace('.png', '_pred_mask.png')),np.uint8(pred_map * 255))


            # output_root = '/home/weishujin/Codes/pytorch/MVSS-Net-master/result_pretrain_crop512_dice_0.0001_wo_sobel'
            # cv2.imwrite(os.path.join(output_root, names[0].split('/')[-1].replace('.jpg', '.png')),(pred_map[0] * 255).astype(np.uint8))
            print('save image:', os.path.join(out, file))
            # for i in range(b):
            #     w, h, c = (cv2.imread(names[i])).shape
            #     cv2.imwrite(os.path.join(output_root,names[i].split('/')[-1].replace('.jpg','.png')),cv2.resize((pred_map[i]*255).astype(np.uint8),(h,w)))
            #     print('save image:',os.path.join(output_root,names[i].split('/')[-1].replace('.jpg','.png')))



def test(test_loader, mode='joint', visual='N'):
    model.eval()
    with torch.no_grad():
        epoch_loss, correct, total, f1, iou, fpr = 0, 0, 0, 0, 0, 0
        for step, data in enumerate(test_loader):
            images, labels, _, names = data
            images, labels = images.to(device), labels.to(device)
            b, h, w, c = labels.shape
            labels = labels.long()
            # images = images.permute([0, 3, 1, 2])
            if mode == 'joint':
                assert b==1, 'Plese set batch size 1'
                images = images.permute([0, 2, 3, 1])
                images = images.cpu().detach().numpy()
                images, row, col, img = read_image_block_without_mask(images, 512, 256)
                images = torch.Tensor(images).to(device)
                images = images.permute([0, 3, 1, 2])
            # _, outputs = model(images,train_mode='test')
            _, outputs = model(images)
            outputs = torch.sigmoid(outputs).to(device)
            if mode == 'joint':
                pred = outputs.permute([0, 2, 3, 1])
                pred = pred.cpu().detach().numpy()
                pred = pred.squeeze(-1)
                pred = consist_whole_mask(pred, row, col, step=256, IMG_SIZE=512)
                pred = (torch.Tensor(pred)).to(device)
            else:
                pred = outputs.permute([0, 2, 3, 1])
            loss = criterion(pred, labels)
            epoch_loss += loss.item()
            pred_logits = pred
            pred_map = pred
            pred_map[pred_map >= 0.5] = 1
            pred_map[pred_map < 0.5] = 0
            correct += torch.eq(pred_map, labels).float().mean().item()
            pred_map = pred_map.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            f1_ = f1_score(labels.reshape([-1]).astype('int'), pred_map.reshape([-1]).astype('int'))
            iou_, fpr_ = iou_compute(labels.reshape([-1]).astype('int'), pred_map.reshape([-1]).astype('int'))
            if np.isnan(f1_) == False:
                f1 += f1_
            if np.isnan(iou_) == False:
                iou += iou_
            if np.isnan(fpr_) == False:
                fpr += fpr_
            total += 1
            if visual == 'Y':
                pred_logits = pred_logits.cpu().detach().numpy()
                out = r'/pubdata/weishujin/Deep_inpainting_localization/visual/Trans/MVSS'
                file, imgname = names[0].split('/')[-2], names[0].split('/')[-1]
                if not os.path.exists(os.path.join(out, file, 'pred')): os.makedirs(os.path.join(out, file, 'pred'))
                if not os.path.exists(os.path.join(out, file, 'pred_mask')): os.makedirs(os.path.join(out, file, 'pred_mask'))
                io.imsave(os.path.join(out, file, 'pred', imgname.replace('.png', '_pred.png')), np.uint8(pred_logits[0] * 255))
                io.imsave(os.path.join(out, file, 'pred_mask', imgname.replace('.png', '_pred_mask.png')),np.uint8(pred_map[0] * 255))
                print(file+imgname, end=' ')
                print('Iter [%d/%d], \033[1;31mf1: %.4f\033[0m, iou：%.4f' %(total, len(test_loader), f1_, iou_))
            else:
                sys.stdout.write('->Iter [%d/%d], loss: %.4f, acc: %.4f, \033[1;31mf1: %.4f\033[0m, iou：%.4f, fpr：%.4f'
                             % (total, len(test_loader), epoch_loss / total, correct / total, f1/total, iou/total, fpr/total))
                sys.stdout.write('\n')

        return round(epoch_loss/total, 3), round(correct/total, 3), round(f1/total, 3), round(iou/total, 3),  round(fpr/total, 3)

def train(train_loader, valid_loader, optimizer, current_epoch, print_func):
    print('Start training...')
    best_f1 = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)
    for epoch in range(args.Epoch)[current_epoch:]:
        device = torch.device('cuda')
        epoch_loss, epoch_out_loss, epoch_edg_loss, correct, total = 0, 0, 0, 0, 0
        for step, data in enumerate(train_loader):
            star_time = time.time()
            images, labels, edg_labels, _ = data
            images, labels = images.to(device), labels.to(device)
            b, h, w, c = labels.shape
            labels = labels.reshape([-1, h, w]).long()
            optimizer.zero_grad()
            res1,outputs = model(images)
            outputs = torch.sigmoid(outputs)
            pred = outputs.permute([0,2,3,1])
            if args.use_sobel == True:
                edg_labels = (edg_labels.to(device)).reshape([-1, h//4, w//4]).long()
                res1 = torch.sigmoid(res1)
                res1 = res1.permute([0, 2, 3, 1])
                out_loss = criterion(pred, labels)
                edg_loss = criterion(res1, edg_labels)
                epoch_out_loss += out_loss.item()
                epoch_edg_loss += edg_loss.item()
                loss = args.a_lamda * out_loss + args.b_lamda * edg_loss
            else:
                loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pred_map = outputs
            pred_map[pred_map >= 0.5] = 1.
            pred_map[pred_map < 0.5] = 0.
            correct += torch.eq(pred_map, labels).float().mean().item()
            total += 1

            end_time = time.time()
            present_hour,present_min, total_day, total_hour,total_min = time_calucate(star_time,end_time,epoch+1,args.Epoch,step,len(train_loader))
            if args.use_sobel == True:
                sys.stdout.write('\r->Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, out_loss: %.4f, edg_loss: %.4f, Acc: %.4f, Lr：%.1e, Present_Time：%02d:%02d, Total_Time：%01d天%02d小时%02d分钟'
                                 % (epoch + 1, args.Epoch, total, len(train_loader), epoch_loss/total,epoch_out_loss/total, epoch_edg_loss/total, correct/total, optimizer.param_groups[0]['lr'],
                                    present_hour, present_min,total_day, total_hour,total_min))
            else:
                sys.stdout.write(
                    '\r->Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, Acc: %.4f, Lr：%.1e, Present_Time：%02d:%02d, Total_Time：%01d天%02d小时%02d分钟'
                    % (epoch + 1, args.Epoch, total, len(train_loader), epoch_loss / total, correct / total,
                       optimizer.param_groups[0]['lr'], present_hour, present_min, total_day, total_hour, total_min))
            if step % (len(train_loader)//1) == 0 and step != 0:
                print('')
                print_func('->Epoch [%d/%d], Iter [%d/%d], Loss: %.4f, Acc: %.4f'% (epoch + 1, args.Epoch, total, len(train_loader), epoch_loss/total, correct/total))

        print('\nWaiting val...')
        val_loss, val_acc, val_f1, val_iou, _ = test(valid_loader, mode='origin')
        if best_f1 < val_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path + '/' + str(epoch + 1) + '_acc_' + str(val_acc) + '_f1_' + str(val_f1) + '_lr_' + str(args.lr) + '.pkl')
        print('')
        print_func('->Epoch [%d/%d], val_loss: %.4f, val_acc: %.3f, \033[1;31mval_f1: %.3f\033[0m, val_iou: %.3f'
                     % (epoch + 1, args.Epoch, val_loss, val_acc, val_f1, val_iou))
        scheduler.step()

def main():
    if args.mode == 'train':
        write_log_mode = 'w'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        else:
            if os.listdir(save_path):
                sys.stderr.write('Log dir is not empty, continue? [yes(y)/remove(r)/no(n)]: ')
                chioce = input('')
                if (chioce == 'y' or chioce == 'Y'):
                    write_log_mode = 'a'
                elif (chioce == 'r' or chioce == 'R'):
                    rmtree(save_path)
                else:
                    sys.stderr.write('Abort.\n')
                    return None
        tee_print = TeePrint(filename=save_path + 'train.log', mode=write_log_mode)
        print_func = tee_print.write
    else:
        print_func = print
    print_func('--------------Configuration--------------')
    print_func('Epoch: %s\nlr: %s\nbatch_size: %s\nlr_decay：%.3f\nmode: %s\nloss: %s\nsave_path: %s'
    % (str(args.Epoch), str(args.lr), str(args.batch_size), args.lr_decay, args.mode, args.loss, save_path))
    print_func('-----------------------------------------')
    print(test_root)

    if args.mode == 'train':
        if args.restore_path == 'retrain':
            args.restore_path = glob.glob(os.path.join(save_path, '*'))
            args.restore_path.sort()
            args.restore_path = args.restore_path[-1]
            model.load_state_dict(torch.load(args.restore_path))
            current_epoch = int(args.restore_path.split('/')[-1].split('_')[0])
            args.lr = args.lr * (0.5**(current_epoch-1))
            print('load model：%s' % args.restore_path)
        else:
            current_epoch = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # # docimg png
        # train_data = read_dataset(args, train_root, mode='train', pattern='*.png',print_func=print_func)
        # valid_data = read_dataset(args, valid_root, mode='val', pattern='*.png',print_func=print_func)

        # Alinew
        train_data = read_dataset(args, train_root, mode='train', pattern='*.jpg',print_func=print_func)
        valid_data = read_dataset(args, valid_root, mode='val', pattern='*.jpg',print_func=print_func)

        train_imgs, train_labels = train_data
        # num = (len(train_imgs) // args.batch_size) * args.batch_size
        # train_imgs, train_labels = train_imgs[:num], train_labels[:num]
        valid_imgs, valid_labels = valid_data

        # fnum = 16
        # train_imgs = train_imgs[:fnum]
        # train_labels = train_labels[:fnum]
        # valid_imgs = valid_imgs[:2]
        # valid_labels = valid_labels[:2]

        train_data = MyDataset(imgs=train_imgs, labels=train_labels, use_sobel=args.use_sobel,resize=resize)
        valid_data = MyDataset(imgs=valid_imgs, labels=valid_labels, use_sobel=args.use_sobel,resize=resize)

        train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)

        train(train_loader, valid_loader, optimizer, current_epoch, print_func=print_func)

    elif args.mode in ['test', 'visual']:
        test_data = read_dataset(args, test_root, mode=args.mode, pattern='*.jpg', print_func=print_func)
        test_imgs, test_labels = test_data
        test_data = MyDataset(imgs=test_imgs, labels=test_labels,resize=resize, use_sobel=args.use_sobel)
        test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
        # model.load_state_dict(torch.load(args.restore_path))
        # print('load model：%s' % args.restore_path)
        args.restore_path = glob.glob(os.path.join(save_path, '*'))
        args.restore_path.sort()
        args.restore_path = args.restore_path[-1]
        model.load_state_dict(torch.load(args.restore_path))
        if args.mode == 'visual':
            test_loss, test_acc, test_f1, test_iou, test_fpr = test(test_loader, mode='r', visual='Y')
        else:
            test_loss, test_acc, test_f1, test_iou, test_fpr = test(test_loader,mode='r')
        print_func('\ntest_loss: %.3f\ntest_acc: %.3f\n\033[1;31mtest_f1: %.3f\033[1;31m\ntest_iou: %.3f\ntest_fpr: %.3f' % (
        test_loss, test_acc, test_f1, test_iou, test_fpr))

        # else:
        #     args.restore_path = glob.glob(os.path.join(save_path, '*'))
        #     args.restore_path.sort()
        #     args.restore_path = args.restore_path[-1]
        #     model.load_state_dict(torch.load(args.restore_path))
        #     print('load model：%s' % args.restore_path)
        #     visual(test_loader, mode='origin')


# conda activate pt
# cd /home/zhengkengtao/codes/MVSS-Net/
# python train_Alinew_ft.py
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 5'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Epoch", type=int, default=300)
    parser.add_argument("--lr",    type=float, default=1e-4)
    parser.add_argument("--a_lamda", type=float, default=0.2)
    parser.add_argument("--b_lamda", type=float, default=0.8)
    parser.add_argument("--use_sobel", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=2*4)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--loss", type=str, default='dice')
    # parser.add_argument("--data_root", type=str, default='/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/train_imgs')
    parser.add_argument("--data_root", type=str, default='/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/train_imgs')
    # parser.add_argument("--data_root", type=str, default='/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/train_imgs')
    # parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--restore_path', type=str, default='/data1/zhengkengtao/exps/0723_MVSSNet_docimg_split811_png/63_acc_0.96_f1_0.707_lr_0.0001.pkl')
    args = parser.parse_args()

    device = torch.device('cuda')

    mvssnet_path = None
    model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=1,
                     sobel=args.use_sobel,
                     constrain=True,
                     n_input=3,
                     )
    model = nn.DataParallel(model)
    model.to(device)
    if mvssnet_path is not None:
        checkpoint = torch.load(mvssnet_path)
        model.load_state_dict(checkpoint, strict=True)
    if args.loss == 'dice':
        criterion = DiceLoss().to(device)

    train_root = [args.data_root]
    # valid_root = ['/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split_1000_200_2800/val_imgs']
    valid_root = ['/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split118/val_imgs']
    # valid_root = ['/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/test_imgs']
    test_root = None
    print(train_root)
    print(valid_root)
    print(test_root)

    train_resize = (768, 768) # docimg (512, 512) Alinew (768, 768) None
    resize = train_resize
    print('resize:', resize)

    save_path = os.path.join('/data1/zhengkengtao/exps/1617_MVSS_Alinew_trainsplit118_noAug_pretrainwithdocimgsplit811Aug_1/')
    main()
























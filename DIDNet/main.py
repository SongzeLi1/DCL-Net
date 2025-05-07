# conda activate pt
# cd /home/zhengkengtao/codes/DIDNet/
# python main.py

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from numpy.lib.function_base import corrcoef
from utils import *
from models import *
import argparse
from time import time
import pandas as pd
from random import shuffle
from tqdm import tqdm
import warnings
from PIL import ImageFile
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=90)
    parser.add_argument('--lr', type=float, default=0.0003, help='initial learning rate')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--step_size', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--save_result_dir', type=str, default='/data1/zhengkengtao/exps/1023_DIDNet_docimg_split811_png_64x64_OrigMosaicOneKind/')
    parser.add_argument('--pretrain_ckpt', type=str, default=None)
    # parser.add_argument('--pretrain_ckpt', type=str, default='/data1/zhengkengtao/exps/0714_DIDNet_docimg_split811_png_64x64/DIDNet_epoch_86.pth')
    return parser.parse_args()

def test(Test_loader, model, cfg):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(Test_loader):
            image, srm, label = data
            image, srm, label =  image.to(cfg.device), srm.to(cfg.device), label.to(cfg.device)
            output = model(image, srm)
            _, predict = torch.max(output.data, dim = 1) 
            total += label.size(0)
            correct += (predict == label).sum().item()
    print('accuracy on Test set: %f %%' % float(100 * correct / total))

def val(val_loader, model, cfg):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in tqdm(val_loader):
            image, srm, label = data
            image, srm, label =  image.to(cfg.device), srm.to(cfg.device), label.to(cfg.device)
            output = model(image, srm)
            _, predict = torch.max(output.data, dim = 1)
            total += label.size(0)
            correct += (predict == label).sum().item()
    print('accuracy on val set: %f %%' % float(100 * correct / total))
    return float(100*correct/total)


def train(train_loader, val_loader, cfg, test_loader = 0):
    # model = DIDNet().to(cfg.device)
    # model = nn.DataParallel(model, device_ids = [0,1,2,3])
    model = DIDNet()
    model = nn.DataParallel(model)
    model.cuda()
    # ---加载预训练模型---
    if cfg.pretrain_ckpt is not None:
        print('---load checkpoint: {} ---'.format(cfg.pretrain_ckpt))
        checkpoint = torch.load(cfg.pretrain_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
    # ------------------
    model.apply(init_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr, weight_decay = cfg.decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = cfg.step_size, gamma = cfg.gamma, verbose = False)
    lossEpoch = []
    lossList = []
    valACCs = []
    maxValAcc = 0.0

    for epoch in range(cfg.n_epochs):
        time_start = time()
        model.train()
        lossEpoch.clear()
        for i, data in enumerate(tqdm(train_loader)):
            image, srm, label = data
            image, srm, label =  image.to(cfg.device), srm.to(cfg.device), label.to(cfg.device)
            output = model(image, srm)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossEpoch.append(loss.item())
        lossList.append(float(np.array(lossEpoch).mean()))
        print("epoch:",str(epoch),"trainingloss:",lossList[epoch])
        scheduler.step()
        valACC = val(val_loader, model, cfg)
        valACCs.append(valACC)
        if valACC > maxValAcc:
            maxValAcc = valACC
            torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    }, cfg.save_result_dir + 'DIDNet_epoch_' + str(epoch)+'.pth')
        test(test_loader, model, cfg)
        time_end = time()
        print("time:" + str(time_end - time_start))
    return valACCs.index(max(valACCs))

def main(cfg):
    if not os.path.exists(cfg.save_result_dir): os.makedirs(cfg.save_result_dir)
    # Load train, val, test dataset

    # # ### 172.31.224.53 docimg
    # train_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/train_tamper/', mode="tamper")
    # lenght = len(train_list_tamper)
    # train_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/train_real/', mode="real", subsample=lenght)
    # print(lenght)
    # val_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/val_tamper/', mode="tamper")
    # lenght = len(val_list_tamper)
    # val_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/val_real/', mode="real", subsample=lenght)
    # print(lenght)
    # test_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/test_tamper/', mode="tamper")
    # lenght = len(test_list_tamper)
    # test_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select/test_real/', mode="real", subsample=lenght)
    # print(lenght)

    # ### 172.31.224.53 docimg Orig和Mosaic视为一类，Tamper为另一类
    train_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/train_tamper/', mode="tamper")
    lenght = len(train_list_tamper)
    train_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/train_real/', mode="real", subsample=lenght)
    print(lenght)
    val_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/val_tamper/', mode="tamper")
    lenght = len(val_list_tamper)
    val_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/val_real/', mode="real", subsample=lenght)
    print(lenght)
    test_list_tamper = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/test_tamper/', mode="tamper")
    lenght = len(test_list_tamper)
    test_list_real = read_list('/data1/zhengkengtao/docimg/docimg_split811/crop_64x64/select_OrigMosaicOneKind/test_real/', mode="real", subsample=lenght)
    print(lenght)

    # ### 172.31.224.53 Alinew
    # datas_dir = '/data1/zhengkengtao/Ali_new/forgery_round1_train_20220217/train_split19/crop_64x64/select/'
    # train_list_tamper = read_list(datas_dir + 'train_tamper/', mode="tamper")
    # lenght = len(train_list_tamper)
    # train_list_real = read_list(datas_dir + 'train_real/', mode="real", subsample=lenght)
    # print(lenght)
    # val_list_tamper = read_list(datas_dir + 'test_tamper/', mode="tamper")
    # lenght = len(val_list_tamper)
    # val_list_real = read_list(datas_dir + 'test_real/', mode="real", subsample=lenght)
    # print(lenght)
    # test_list_tamper = read_list(datas_dir + 'test_tamper/', mode="tamper")
    # lenght = len(test_list_tamper)
    # test_list_real = read_list(datas_dir + 'test_real/', mode="real", subsample=lenght)
    # print(lenght)
    
    train_list = train_list_real + train_list_tamper
    val_list =  val_list_real + val_list_tamper
    test_list = test_list_real + test_list_tamper

    train_label_list = np.ones(len(train_list))
    train_label_list[:int(len(train_list) / 2)] = 0
    val_label_list = np.ones(len(val_list))
    val_label_list[:int(len(val_list) / 2)] = 0

    test_label_list = np.ones(len(test_list))
    test_label_list[:int(len(test_list) / 2)] = 0


    train_set = DataSetLoader(train_list, train_label_list)
    train_loader = DataLoader(dataset = train_set, num_workers = 8, batch_size=cfg.batch_size, shuffle=True)

    val_set = DataSetLoader(val_list, val_label_list)
    val_loader = DataLoader(dataset=val_set, num_workers=8, batch_size=cfg.batch_size, shuffle=False)

    test_set = DataSetLoader(test_list, test_label_list)
    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=cfg.batch_size, shuffle=False)

    ## Training model
    maxIndex = train(train_loader, val_loader, cfg, test_loader)
    print('best epoch:', maxIndex)

    # modelTest = DIDNet().to(cfg.device)
    # modelTest = nn.DataParallel(modelTest, device_ids = [0,1])
    # pretrained_dict = torch.load('model_ali/DIDNet_epoch_'+str(maxIndex)+'.pth')
    # modelTest.load_state_dict(pretrained_dict['state_dict'])
    # test(test_loader, modelTest, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)





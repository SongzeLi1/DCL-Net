import os
import tempfile

import cv2
import jpegio
import numpy as np
import torch
import torchvision
from PIL import Image
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from models.dtd import seg_dtd

device=torch.device("cuda")
# model = seg_dtd("",2).to(device)

data_path = '/pubdata/lisongze/docimg/exam/docimg2jpeg/val_images_90/'
new_qtb = np.array(
    [[2, 1, 1, 2, 2, 4, 5, 6], [1, 1, 1, 2, 3, 6, 6, 6], [1, 1, 2, 2, 4, 6, 7, 6], [1, 2, 2, 3, 5, 9, 8, 6],
     [2, 2, 4, 6, 7, 11, 10, 8], [2, 4, 6, 6, 8, 10, 11, 9], [5, 6, 8, 9, 10, 12, 12, 10],
     [7, 9, 10, 10, 11, 10, 10, 10]], dtype=np.int32).reshape(64, ).tolist()
totsr = ToTensorV2()
toctsr =torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))
    ])


for path in tqdm(os.listdir(data_path)):
    if str(path).endswith("jpg"):
        img_path = os.path.join(data_path, path)
        imgs_ori = cv2.imread(img_path)
        h, w, c = imgs_ori.shape
        jpg_dct = jpegio.read(img_path)
        dct_ori = jpg_dct.coef_arrays[0].copy()
        use_qtb2 = jpg_dct.quant_tables[0].copy()

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            imgs_ori = Image.fromarray(imgs_ori).convert("L")
            imgs_ori.save(tmp, "JPEG", qtables={0: new_qtb})
            jpg = jpegio.read(tmp.name)
            dct_ori = jpg.coef_arrays[0].copy()
            imgs_ori = np.array(imgs_ori.convert('RGB'))
            use_qtb2 = jpg.quant_tables[0].copy()

        if h % 8 == 0 and w % 8 == 0:
            imgs_d = imgs_ori
            dct_d = dct_ori
        else:
            imgs_d = imgs_ori[0:(h // 8) * 8, 0:(w // 8) * 8, :].copy()
            dct_d = dct_ori[0:(h // 8) * 8, 0:(w // 8) * 8].copy()

        qs = torch.LongTensor(use_qtb2)
        img_h, img_w, _ = imgs_d.shape
        img_list = []
        for idx, crop in enumerate(imgs_d):
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            data = toctsr(crop)
            dct = torch.LongTensor(dct_d[idx])

            data, dct, qs = data.unsqueeze(0).to(device), dct.unsqueeze(0).to(device), qs.unsqueeze(0).to(device)
            dct = torch.abs(dct).clamp(0, 20)
            B, C, H, W = data.shape
            qs = qs.reshape(B, 1, 8, 8)
        #     with torch.no_grad():
        #         if data.size()[-2:] == torch.Size((512, 512)) and dct.size()[-2:] == torch.Size(
        #                 (512, 512)) and qs.size()[-2:] == torch.Size((8, 8)):
        #             pred = model(data, dct, qs)
        #             pred = torch.nn.functional.softmax(pred, 1)[:, 1].cpu()
        #             img_list.append(((pred.cpu().numpy()) * 255).astype(np.uint8))
        # padding = (0, 0, w - img_w, h - img_h)
        # ci = cv2.copyMakeBorder(img_list, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT,
        #                         value=[0, 0, 0])
        # cv2.imwrite(os.path.join(os.path.join(data_path, 'docimg2jpeg'), path), ci)
            print(data,dct,qs)

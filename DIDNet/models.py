# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.8.10 (default, Jun  4 2021, 15:09:15) 
# [GCC 7.5.0]
# Embedded file name: ./models.py
# Compiled at: 2022-02-28 16:03:18
# Size of source mod 2**32: 7838 bytes
import torch, torch.nn as nn, torch.nn.functional as F
from torchsummary import summary

class BN_Conv2d(nn.Module):
    __doc__ = '\n    BN_CONV_RELU\n    '

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
          dilation=dilation,
          groups=groups,
          bias=bias), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return F.relu(self.seq(x))


class Inception_A(nn.Module):
    __doc__ = '\n    Inception-A block for Inception-v4 net\n    '

    def __init__(self, in_channels, b1, b2, b3_n1, b3_n3, b4_n1, b4_n3):
        super(Inception_A, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d(3, 1, 1), BN_Conv2d(in_channels, b1, 1, 1, 0, bias=False))
        self.branch2 = BN_Conv2d(in_channels, b2, 1, 1, 0, bias=False)
        self.branch3 = nn.Sequential(BN_Conv2d(in_channels, b3_n1, 1, 1, 0, bias=False), BN_Conv2d(b3_n1, b3_n3, 3, 1, 1, bias=False))
        self.branch4 = nn.Sequential(BN_Conv2d(in_channels, b4_n1, 1, 1, 0, bias=False), BN_Conv2d(b4_n1, b4_n3, 3, 1, 1, bias=False), BN_Conv2d(b4_n3, b4_n3, 3, 1, 1, bias=False))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat((out1, out2, out3, out4), 1)


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):

    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.LeakyReLU(0.1, inplace=False)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        else:
            for i in range(reps - 1):
                rep.append(self.relu)
                rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(filters))

            if not grow_first:
                rep.append(self.relu)
                rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
                rep.append(nn.BatchNorm2d(out_filters))
            if not start_with_relu:
                rep = rep[1:]
            else:
                rep[0] = nn.LeakyReLU(0.1, inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(kernel_size=2, stride=strides))
        self.rep = (nn.Sequential)(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class DIDNet(nn.Module):
    expansion = 1

    def __init__(self):
        super(DIDNet, self).__init__()
        self.conv_rgb_1 = Inception_A(3, 16, 16, 8, 16, 8, 16)
        self.conv_rgb_1_transition = nn.Sequential(nn.Conv2d(64, 16, kernel_size=1, stride=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=False))
        self.conv_rgb_2 = Inception_A(16, 32, 32, 24, 32, 24, 32)
        self.conv_rgb_2_transition = nn.Sequential(nn.Conv2d(128, 32, kernel_size=1, stride=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=False))
        self.conv_rgb_3 = Inception_A(32, 64, 64, 48, 64, 48, 64)
        self.conv_rgb_3_transition = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1, stride=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=False))
        self.conv_srm_1_1 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=False))
        self.conv_srm_1_2 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=False))
        self.conv_srm_2_1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=False))
        self.conv_srm_2_2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=False))
        self.conv_srm_3_1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=False))
        self.conv_srm_3_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=False))
        self.conv4 = Block(176, 256, 2, 2, start_with_relu=False, grow_first=True)
        self.conv5 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.conv6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.conv7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=4, mode='nearest')
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(728, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, srm):
        batch_size = srm.size(0)
        feature_spatial = self.conv_rgb_1(image)
        feature_spatial = self.conv_rgb_1_transition(feature_spatial)
        feature_spatial = self.conv_rgb_2(feature_spatial)
        feature_spatial = self.conv_rgb_2_transition(feature_spatial)
        feature_spatial = self.conv_rgb_3(feature_spatial)
        feature_spatial = self.conv_rgb_3_transition(feature_spatial)
        srm_1 = self.conv_srm_1_1(srm)
        srm_1 = self.conv_srm_1_2(srm_1)
        srm_1_pool = self.max_pool(srm_1)
        srm_2 = self.conv_srm_2_1(srm_1_pool)
        srm_2 = self.conv_srm_2_2(srm_2)
        srm_2_pool = self.max_pool(srm_2)
        srm_3 = self.conv_srm_3_1(srm_2_pool)
        srm_3 = self.conv_srm_3_2(srm_3)
        srm_2_upSample = self.up1(srm_2)
        srm_3_upSample = self.up2(srm_3)
        featureCat = torch.cat([feature_spatial, srm_1, srm_2_upSample, srm_3_upSample], 1)
        featureCat = self.conv4(featureCat)
        featureCat = self.conv5(featureCat)
        featureCat = self.conv6(featureCat)
        featureCat = self.conv7(featureCat)
        y = self.globalAvgPool(featureCat)
        y = torch.flatten(y, 1)
        y = self.fc1(y)
        y = self.softmax(y)
        return y


if __name__ == '__main__':
    model = DIDNet().to('cuda:0')
    summary(model, input_size=(3, 320, 320), device='cuda')
# okay decompiling models.pyc

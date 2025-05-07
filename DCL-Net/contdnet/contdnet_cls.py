from .convnext import *
import torch.nn as nn
import torch
import warnings
from PIL import ImageFile
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True


def freeze_bn(m):
    """
    https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/12
    """
    if isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight'):
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias'):
            m.bias.requires_grad_(False)
        m.eval()  # for freeze bn layer's parameters 'running_mean' and 'running_var


# from mmcv.cnn import constant_init, kaiming_init
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class Upsample(nn.Module):

    def __init__(self, factor=2) -> None:
        super(Upsample, self).__init__()
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode="bilinear")


class ConvBnAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, momentum=0.001)  # if use gradient accumulation then set a small momentum of BN
        self.act = nn.SiLU(inplace=True) if act else nn.Identity() # SiLU=x∗sigmoid(x)

    def forward(self, x):
        x = self.conv(x)
        # print('before bn:', x.shape)
        x = self.bn(x)  # 有bn所以batch size不能为1，注释掉此句就可以为1
        # print('after bn:', x.shape)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels // 4
        self.cba1 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba2 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba3 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba4 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.out = ConvBnAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.cba1(self.pool(x, 1)), size)
        f2 = self.upsample(self.cba2(self.pool(x, 2)), size)
        f3 = self.upsample(self.cba3(self.pool(x, 3)), size)
        f4 = self.upsample(self.cba4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)


# ConTDNet_cls
class ConTDNet_cls(nn.Sequential):
    def __init__(self):
        super(ConTDNet_cls, self).__init__()
        self.convnext = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)
        self.num_classes = 3
        self.fpn_dim = 384
        self.ppm = PyramidPoolingModule(768, self.fpn_dim)
        self.cls = nn.Linear(384, 1)
    def forward(self, x, args3=None, args4=None):
        [x1, x2, x3, x4], _, _ = self.convnext(x)
        out_dict = {}
        out_dict['convnext_layer1'] = x1
        out_dict['convnext_layer2'] = x2
        out_dict['convnext_layer3'] = x3
        out_dict['convnext_layer4'] = x4
        ppm_f = self.ppm(out_dict['convnext_layer4'])
        ppm_f_mean = ppm_f.mean([-2, -1])
        cls = self.cls(ppm_f_mean)
        return cls, ppm_f















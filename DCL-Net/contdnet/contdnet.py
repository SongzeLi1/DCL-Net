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


class FeaturePyramidNet(nn.Module):

    def __init__(self, fpn_dim=512):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.fpn_in = nn.ModuleDict({'fpn_layer1': ConvBnAct(self.fpn_dim // 4, self.fpn_dim, 1, 1, 0),
                                     "fpn_layer2": ConvBnAct(self.fpn_dim // 2, self.fpn_dim, 1, 1, 0),
                                     "fpn_layer3": ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0),
                                     })
        self.fpn_out = nn.ModuleDict({'fpn_layer1': ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1),
                                      "fpn_layer2": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1),
                                      "fpn_layer3": ConvBnAct(self.fpn_dim, self.fpn_dim, 3, 1, 1),
                                      })

    def forward(self, pyramid_features):

        fpn_out = {}

        f = pyramid_features['convnext_layer4']
        fpn_out['fpn_layer4'] = f

        x = self.fpn_in['fpn_layer3'](pyramid_features['convnext_layer3'])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer3'] = self.fpn_out['fpn_layer3'](f)

        x = self.fpn_in['fpn_layer2'](pyramid_features['convnext_layer2'])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer2'] = self.fpn_out['fpn_layer2'](f)

        x = self.fpn_in['fpn_layer1'](pyramid_features['convnext_layer1'])
        f = F.interpolate(f, x.shape[2:], mode='bilinear', align_corners=False)
        f = x + f
        fpn_out['fpn_layer1'] = self.fpn_out['fpn_layer1'](f)

        return fpn_out


class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool='att', fusions=['channel_add'], ratio=8):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out


class ConTDNet(nn.Sequential):
    def __init__(self):
        super(ConTDNet, self).__init__()
        self.convnext = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)
        self.num_classes = 3
        self.fpn_dim = 384
        self.ppm = PyramidPoolingModule(768, self.fpn_dim)
        self.gcAtt = ContextBlock2d(inplanes=4*384, planes=4*384)
        self.fpn = FeaturePyramidNet(self.fpn_dim)
        self.fuse = ConvBnAct(self.fpn_dim * 4, self.fpn_dim, 1, 1, 0)
        self.conv1 = ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0)
        self.seg = nn.Conv2d(self.fpn_dim, 3, 1, 1, 0, bias=True)
        self.out = nn.Conv2d(self.num_classes, self.num_classes, 3, 1, 1)
    def forward(self, x):
        [x1, x2, x3, x4], _, _ = self.convnext(x)
        out_dict = {}
        out_dict['convnext_layer1'] = x1
        out_dict['convnext_layer2'] = x2
        out_dict['convnext_layer3'] = x3
        out_dict['convnext_layer4'] = x4
        ppm_f = self.ppm(out_dict['convnext_layer4'])
        out_dict.update({'convnext_layer4': ppm_f})
        fpn_f = self.fpn(out_dict)
        out_size = fpn_f['fpn_layer1'].shape[2:]
        list_f = []
        list_f.append(fpn_f['fpn_layer1'])
        list_f.append(F.interpolate(fpn_f['fpn_layer2'], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn_f['fpn_layer3'], out_size, mode='bilinear', align_corners=False))
        list_f.append(F.interpolate(fpn_f['fpn_layer4'], out_size, mode='bilinear', align_corners=False))
        res = torch.cat(list_f, dim=1)
        res = self.gcAtt(res)  # 增加
        res = self.fuse(res)
        res = self.conv1(res)
        res = self.seg(res)
        res = F.interpolate(res, x.shape[2:], mode='bilinear', align_corners=False)
        out = self.out(res)
        return out, ppm_f















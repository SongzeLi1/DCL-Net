import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam
from torchvision.models.resnet import BasicBlock,ResNet
from torchvision.models.densenet import _densenet
import numpy as np
# from networks import highPassingFilters
# import pytorch_colors as colors #pytorch颜色空间转换包
# 2xpadding-dilation*(k-1) = 0
class ConvX(nn.Module):
    def __init__(self,in_channels,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training):
        super(ConvX,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, stride=strides, padding=padding,dilation=dilate_rate)
        self.bn_in = bn_in
        if (self.bn_in == 'bn'):
            self.bn_layer = nn.BatchNorm2d(num_features=filters,affine=True)
        if (self.bn_in == 'in'):
            self.in_layer = nn.InstanceNorm2d(num_features=filters,affine = True)
        self.act_layer = nn.ReLU(inplace=True) #不保存中间变量

    def forward(self,x):
        x = self.conv(x)
        if(self.bn_in =='bn'):
            x = self.bn_layer(x)
        if(self.bn_in == 'in'):
            x = self.in_layer(x)
        x = self.act_layer(x)
        return x

class dense_block(nn.Module):
    def __init__(self,in_channels,num_conv,kernel_size,filters,output_channels,dilate_rate,weight_decay,name,down_sample,is_training,bn_in,strides,padding):
        super(dense_block, self).__init__()
        self.num_conv = num_conv
        # if(self.num_conv==2):
        #     self.conv1 = ConvX(in_channels,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        #     self.conv2 = ConvX(in_channels+filters,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        # if(self.num_conv==4):
        #     self.conv1 = ConvX(in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
        #                        is_training)
        #     self.conv2 = ConvX(in_channels + filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
        #                        dilate_rate, is_training)
        #     self.conv3 = ConvX(in_channels+2*filters,filters,kernel_size,strides,padding,weight_decay,bn_in,dilate_rate,is_training)
        #     self.conv4 = ConvX(in_channels + 3 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,dilate_rate, is_training)
        self.conv1 = ConvX(in_channels, filters, kernel_size, strides, padding, weight_decay, bn_in, dilate_rate,
                           is_training)
        self.conv2 = ConvX(in_channels + filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.conv3 = ConvX(in_channels + 2 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.conv4 = ConvX(in_channels + 3 * filters, filters, kernel_size, strides, padding, weight_decay, bn_in,
                           dilate_rate, is_training)
        self.down_sample = down_sample
        if(self.num_conv==2):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 2 * filters, out_channels=output_channels, kernel_size=kernel_size, stride=strides, padding=padding,dilation=dilate_rate)
        if (self.num_conv == 3):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 3 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
        if (self.num_conv == 4):
            self.transition_layer = nn.Conv2d(in_channels=in_channels + 4 * filters, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=strides, padding=padding,
                                              dilation=dilate_rate)
    def forward(self,x):
        if(self.num_conv==2):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x,conv1_output],dim = 1)
            conv2_output = self.conv2(conv2_input)
            transition_input = torch.cat([x,conv1_output,conv2_output],dim = 1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
                # x = nn.AvgPool2d(kernel_size=2, stride=2, padding='same')(x)
            return x
        if (self.num_conv == 3):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x, conv1_output], dim=1)
            conv2_output = self.conv2(conv2_input)
            conv3_input = torch.cat([x, conv1_output, conv2_output], dim=1)
            conv3_output = self.conv3(conv3_input)
            transition_input = torch.cat([x, conv1_output, conv2_output, conv3_output], dim=1)
            x = self.transition_layer(transition_input)
            if (self.down_sample == True):
                x = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(x)
            return x
        if(self.num_conv==4):
            conv1_output = self.conv1(x)
            conv2_input = torch.cat([x,conv1_output],dim = 1)
            conv2_output = self.conv2(conv2_input)
            conv3_input = torch.cat([x,conv1_output,conv2_output],dim = 1)
            conv3_output = self.conv3(conv3_input)
            conv4_input = torch.cat([x,conv1_output,conv2_output,conv3_output],dim = 1)
            conv4_output = self.conv4(conv4_input)
            transition_input = torch.cat([x,conv1_output,conv2_output,conv3_output,conv4_output],dim = 1)
            x = self.transition_layer(transition_input)
            if(self.down_sample==True):
                x = nn.AvgPool2d(kernel_size=2,stride = 2,padding=0)(x)
            return x


def make_DCT_filter_anysize(win_size):
    DCT_filter_n = np.zeros([win_size, win_size, 1, win_size * win_size])

    XX, YY = np.meshgrid(range(win_size), range(win_size))

    C = np.ones(win_size)
    C[0] = np.sqrt(1 / win_size)
    C[1:] = np.sqrt(2 / win_size)
    for v in range(win_size):
        for u in range(win_size):
            DCT_filter_n[:, :, 0, u + v * win_size] = C[v] * C[u] * np.cos(
                (YY + 0.5) * np.pi * v / win_size) * np.cos((XX + 0.5) * np.pi * u / win_size)
    print("DCT_filter_n: ",DCT_filter_n.shape)
    DCT_filter_n = DCT_filter_n.transpose(3,2,0,1)
    DCT_filter = torch.from_numpy(DCT_filter_n.astype(np.float32))

    return DCT_filter

def conv2d_block(x, W, overlapping_step,padding = 0):
    W.requires_grad = False
    return torch.nn.functional.conv2d(x,W,bias = None,stride=(overlapping_step,overlapping_step),padding = 0) #这个在这里存疑

class dct_transform_net(nn.Module):
    def __init__(self,win_size,overlapping_step):
        super(dct_transform_net, self).__init__()
        self.DCT_filter = make_DCT_filter_anysize(win_size)
        self.win_size = win_size
        self.overlapping_step = overlapping_step

    def forward(self,x):
        x = conv2d_block(x,self.DCT_filter,self.overlapping_step)
        return x
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化,output feature：[batch_size,in_channel,1,1]
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x),y.expand_as(x)
class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4,bn_in = 'in'):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)
        if(bn_in=='bn'):
            # print("BNXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        if (bn_in == 'in'):
            # print("INXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(inter_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(channels,affine=True),
            )

            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(inter_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(channels,affine=True),
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        print("xa shape: ",xa.shape,'xl shape: ',xl.shape)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        # print("xo shape: ",xo.shape)
        return xo

# def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
#   if transposed:
#     layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, output_padding=int(stride-1), dilation=dilation, bias=bias)
#     # Bilinear interpolation init 用双线性插值法初始化反卷积核
#     w = torch.Tensor(kernel_size, kernel_size)
#     centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
#     for y in range(kernel_size):
#       for x in range(kernel_size):
#         w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
#     layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
#     return layer
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=stride-1, dilation=dilation, bias=bias)

    # layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=0, dilation=dilation, bias=bias)
    # Bilinear interpolation init 用双线性插值法初始化反卷积核
    w = torch.Tensor(kernel_size, kernel_size)
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  if bias:
    init.constant(layer.bias, 0)
  return layer

class normal_denseFCN_cls(nn.Module):
    def __init__(self, bn_in):
        super(normal_denseFCN_cls, self).__init__()
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dense_block1 = dense_block(in_channels=3, num_conv=4, kernel_size=3, filters=8, output_channels=16,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in=bn_in, strides=1, padding=1)


        self.dense_block2 = dense_block(in_channels=16, num_conv=2, kernel_size=3, filters=16, output_channels=32,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in='in', strides=1, padding=1)

        self.dense_block3 = dense_block(in_channels=32, num_conv=2, kernel_size=3, filters=32, output_channels=64,
                                        dilate_rate=1, weight_decay=0, name='', down_sample=True, is_training=True,
                                        bn_in='in', strides=1, padding=1)

        self.dense_block4 = dense_block(in_channels=64, num_conv=2, kernel_size=3, filters=64, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.dense_block5 = dense_block(in_channels=96, num_conv=2, kernel_size=3, filters=96, output_channels=96,
                                        dilate_rate=3, weight_decay=0, name='', down_sample=False, is_training=True,
                                        bn_in=bn_in, strides=1, padding=3)
        self.cls = nn.Linear(96, 1)

    def forward(self, x):
        processed_image = x
        spatial_input = processed_image
        '''spatial branch'''
        spatial_dense_block1 = self.dense_block1(spatial_input)  # 空域第一个dense block的输出
        spatial_dense_block2 = self.dense_block2(spatial_dense_block1)
        spatial_dense_block3 = self.dense_block3(spatial_dense_block2)
        spatial_dense_block4 = self.dense_block4(spatial_dense_block3)
        spatial_dense_block5 = self.dense_block5(spatial_dense_block4)
        out = spatial_dense_block5.mean([-2, -1])
        out = self.cls(out)
        return out


if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # train_path = '/pubdata/zhengkengtao/docimg/docimg_split811/crop512x512/train_images/'
    # print(len(os.listdir(train_path)))
    x = torch.randn(3, 3, 224, 224).cuda()
    net = normal_denseFCN_cls(bn_in='bn').cuda()
    net = torch.nn.DataParallel(net)
    # net.load_state_dict(torch.load('/pubdata/zhengkengtao/docimg/docimg_split811/train0705_convnexts_MA/difnet_1440x1440/fold_0_best-ckpt.pth'))
    # x = torch.randn(8, 3, 512, 512)
    # net = DIF()
    out = net(x)
    # out, out_w = net(x)
    print('in:', x.shape, x.min(), x.max())
    print('out:', out.shape, out.min(), out.max())
    # print('out_w:', out_w.shape, out_w.min(), out_w.max())
from torch import nn
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import torch

def conv5x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    # 每个stage中扩展的倍数
    extension = 4

    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(inplane, midplane, stride)
        self.bn1 = nn.BatchNorm1d(midplane)
        self.conv2 = conv5x1(midplane, midplane)
        self.bn2 = nn.BatchNorm1d(midplane)
        self.conv3 = conv1x1(midplane, midplane*self.extension)
        self.bn3 = nn.BatchNorm1d(midplane*self.extension)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差数据
        residual = x

        # 卷积
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # 是否直连（如果是Identity block就直连，如果是conv block就对残差边进行卷积）
        if self.downsample != None:
            residual = self.downsample(x)

        # 相加
        out = out + residual
        out = self.relu(out)

        return out

class Basicblock(nn.Module):

    def __init__(self, inplane, midplane, stride, downsample=None):
        super(Basicblock, self).__init__()

        self.conv1 = conv5x1(inplane, midplane, stride)
        self.bn1 = nn.BatchNorm1d(midplane)
        self.conv2 = conv5x1(midplane, midplane)
        self.bn2 = nn.BatchNorm1d(midplane)
        self.relu = nn.ReLU(inplace=False)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 残差数据
        residual = x

        # 卷积
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))

        # 是否直连（如果是Identity block就直连，如果是conv block就对残差边进行卷积）
        if self.downsample != None:
            residual = self.downsample(x)

        # 相加
        out = out + residual
        out = self.relu(out)

        return out

class Decoder_block(nn.Module):

    def __init__(self, inplanes, outplanes, output_padding=0, kernel_size=5, stride=2):
        super(Decoder_block, self).__init__()
        self.upsample = nn.ConvTranspose1d(in_channels=inplanes, out_channels=outplanes,
                           padding=2, kernel_size=kernel_size, stride=stride, bias=False, output_padding=output_padding)
        self.conv1 = conv5x1(inplanes, outplanes)
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.conv2 = conv5x1(outplanes, outplanes)
        self.padding = nn.ConstantPad1d((0,1), 0)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        if x1.shape[2] < x2.shape[2]:
            x1 = self.padding(x1)
        out = torch.cat((x1, x2), dim=1)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class SignalSegNet(nn.Module):

    def __init__(self, block, layers):

        self.inplane = 64

        super(SignalSegNet, self).__init__()

        self.block = block
        self.layers = layers
        self.layers_num = len(self.layers)

        self.conv1 = nn.Conv1d(1, self.inplane, kernel_size=7, stride=3, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)

        # 64, 128, 256, 512是指扩大四倍之前的维度
        # self.stage1 = self.make_layer(self.block, 64, self.layers[0], stride=1)
        # self.stage2 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        # self.stage3 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        # self.stage4 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        # self.stage5 = self.make_layer(self.block, 1024, self.layers[4], stride=2)

        self.stages = nn.ModuleList()
        for i in range(self.layers_num):
            if i == 0:
                s = 1
            else:
                s = 2
            self.stages.append(self.make_layer(self.block, 64*2**i, self.layers[i], stride=s))

        # self.decoder4 = Decoder_block(1024, 512, output_padding=1)
        # self.decoder3 = Decoder_block(512, 256, output_padding=0)
        # self.decoder2 = Decoder_block(256, 128, output_padding=1)
        # self.decoder1 = Decoder_block(128, 64, output_padding=0)

        self.decoders = nn.ModuleList()
        for i in range(self.layers_num, 1, -1):
            if i == 0:
                s = 1
            else:
                s = 2
            self.decoders.append(Decoder_block(64*2**(i-1), 64*2**(i-2)))

        self.unsample_conv = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=3, padding=4, bias=False)
        self.conv_final = conv1x1(32, 2)
        self.bn_final = nn.BatchNorm1d(2)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        downs = []
        for i in range(self.layers_num):
            if i == 0:
                input = out
            else:
                input = downs[i-1]
            downs.append(self.stages[i](input))
        # down1 = self.stage1(out)
        # down2 = self.stage2(down1)
        # down3 = self.stage3(down2)
        # down4 = self.stage4(down3)
        # down5 = self.stage5(down4)
        for i in range(self.layers_num-1):
            if i == 0:
                up1 = self.decoders[i](downs[-1], downs[-2])
            else:
                up1 = self.decoders[i](up1, downs[-i-2])

        # up4 = self.decoder4(down5, down4)
        # up3 = self.decoder3(up4, down3)
        # up2 = self.decoder2(up3, down2)
        # up1 = self.decoder1(up2, down1)

        out = nn.functional.interpolate(input=up1, scale_factor=2, mode='linear', align_corners=True)
        out = self.unsample_conv(out)
        out = self.conv_final(out)
        out = self.bn_final(out)
        out = self.relu_final(out)

        return out
    
    def make_layer(self, block, midplane, block_num, stride=1):
        '''
            block: block模块
            midplane: 每个模块中间的通道维数，一般等于输出维度/4
            block_num: 重复次数
            stride: Conv Block的步长
        '''
        block_list = []

        # 先确定要不要downsamlpe模块
        downsample=None
        if stride!=1 or self.inplane!=midplane:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplane, midplane, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(midplane)
            )
        
        # Conv Block
        conv_block = block(self.inplane, midplane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = midplane

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, midplane, stride=1))
        
        return nn.Sequential(*block_list)
        

if __name__ == "__main__":
    net = SignalSegNet(Basicblock, [2,2])
    input = torch.randn([3,1,10000])
    output = net(x=input)
    print("参数数量：\n", sum(p.numel() for p in net.parameters() if p.requires_grad))
    pass
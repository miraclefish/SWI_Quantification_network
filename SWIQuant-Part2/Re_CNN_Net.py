from torch import nn
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
        self.conv1 = conv5x1(outplanes, outplanes)
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.conv2 = conv5x1(outplanes, outplanes)

    def forward(self, x1):
        out = self.upsample(x1)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Encoder_block(nn.Module):

    def __init__(self, inplanes, outplanes, kernel_size=5, stride=2):
        super(Encoder_block, self).__init__()
        self.downsample = nn.Conv1d(in_channels=inplanes, out_channels=outplanes,
                           padding=2, kernel_size=kernel_size, stride=stride, bias=False)

        self.conv1 = conv5x1(outplanes, outplanes)
        self.bn1 = nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.conv2 = conv5x1(outplanes, outplanes)

    def forward(self, x1):
        out = self.downsample(x1)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class SignalSegNet(nn.Module):

    def __init__(self, block, layers, mode='CNN'):
        
        super(SignalSegNet, self).__init__()

        self.mode = mode # 'U-Net', 'Att', 'Score'
        self.inplane = 64
        self.dim_pra = self.inplane

        self.block = block
        self.layers = layers
        self.layers_num = len(self.layers)

        self.conv1 = nn.Conv1d(1, self.inplane, kernel_size=7, stride=3, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, padding=1, stride=2)

        # 64, 128, 256, 512是指扩大四倍之前的维度
        # n stage 从 64, 128, 256, 512 到 1024

        self.stages = nn.ModuleList()

        if self.mode == 'Res':
            for i in range(self.layers_num):
                if i == 0:
                    s = 1
                else:
                    s = 2
                self.stages.append(self.make_layer(self.block, self.dim_pra*2**i, self.layers[i], stride=s))
        elif self.mode == 'CNN':
            for i in range(self.layers_num):
                if i == 0:
                    s = 1
                    inplanes = self.dim_pra*2**i
                    outplanes = self.dim_pra*2**i
                else:
                    s = 2
                    inplanes = self.dim_pra*2**(i-1)
                    outplanes = self.dim_pra*2**i
                self.stages.append(Encoder_block(inplanes=inplanes, outplanes=outplanes, stride=s))

        self.decoders = nn.ModuleList()
        for i in range(self.layers_num, 1, -1):
            output_padding = 0
            # if i == 2 or i == 3:
            #     output_padding = 1
            # else:
                
            self.decoders.append(Decoder_block(self.dim_pra*2**(i-1), self.dim_pra*2**(i-2), output_padding))

        self.unsample_conv = nn.ConvTranspose1d(self.dim_pra, int(self.dim_pra/2), kernel_size=7, stride=3, padding=1, bias=False)
        self.conv_final = conv1x1(int(self.dim_pra/2), 2)
        self.bn_final = nn.BatchNorm1d(2)
        self.relu_final = nn.ReLU(inplace=True)

    def forward(self, x):
        
        # x.shape = [batch_size, 1, input_size]
        # score.shape = [batch_size, 1, input_size]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        for i in range(self.layers_num):
            out = self.stages[i](out)
            
        # down1 = self.stage1(out)
        # down2 = self.stage2(down1)
        # down3 = self.stage3(down2)
        # down4 = self.stage4(down3)
        # down5 = self.stage5(down4)

        for i in range(self.layers_num-1):
            out = self.decoders[i](out)

        # up4 = self.decoder4(down5)
        # up3 = self.decoder3(up4)
        # up2 = self.decoder2(up3)
        # up1 = self.decoder1(up2)

        out = nn.functional.interpolate(input=out, scale_factor=2, mode='linear', align_corners=True)
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

class ScoreBlock(nn.Module):

    def __init__(self, inplane, midplane, stride):
        super(ScoreBlock, self).__init__()
        
        self.conv1 = conv5x1(inplane, midplane, stride)
        self.bn1 = nn.BatchNorm1d(midplane)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.bn1(self.conv1(x)))
        return out
        

if __name__ == "__main__":
    net = SignalSegNet(Basicblock, [2,2,2,2], mode='CNN')

    input = torch.randn([3,1,5000])
    output = net(x=input)
    print('Inupt shape:', input.shape, ' // Output shape:', output.shape)
    print("参数数量：\n", sum(p.numel() for p in net.parameters() if p.requires_grad))
    pass
import torch
import torch.nn as nn
import  numpy as np
from torch.nn import BatchNorm2d
from  torchvision.models.resnet import BasicBlock
# from model.fusion_acm import AsymBiChaFuse
import torch.nn.functional as F
# from model.utils import init_weights, count_param


class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)


        self.topdown = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels,  kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
        nn.Conv2d(in_channels=self.channels,out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        self.post = nn.Sequential(
        nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(channels,momentum=0.9),
        nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * torch.mul(xl, topdown_wei) + 2 * torch.mul(xh, bottomup_wei)
        xs = self.post(xs)
        return xs


class LightWeightNetwork(nn.Module):
    def __init__(self, in_channels=3, layers=[3, 3, 3], channels=[8, 16, 32, 64], fuse_mode='AsymBi', classes=1,
                 norm_layer=BatchNorm2d,groups=1, norm_kwargs=None, **kwargs):
        super(LightWeightNetwork, self).__init__()
        self.layer_num = len(layers)
        self._norm_layer = norm_layer
        self.groups = groups
        self.momentum=0.9
        stem_width = int(channels[0])  ##channels: 8 16 32 64
        self.stem = nn.Sequential(
        norm_layer(in_channels, momentum=self.momentum),
        nn.Conv2d(in_channels=in_channels,out_channels=stem_width, kernel_size=3, stride=2,padding=1, bias=False),
        norm_layer(stem_width,momentum=self.momentum),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=stem_width,out_channels=stem_width, kernel_size=3, stride=1,padding=1, bias=False),
        norm_layer(stem_width,momentum=self.momentum),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=stem_width,out_channels=stem_width * 2, kernel_size=3, stride=1,padding=1, bias=False),
        norm_layer(stem_width * 2,momentum=self.momentum),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                       out_channels=channels[1],
                                       in_channels=channels[1], stride=1)

        self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                       out_channels=channels[2], stride=2,
                                       in_channels=channels[1])
        #
        self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                       out_channels=channels[3], stride=2,
                                       in_channels=channels[2])

        self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3] ,out_channels=channels[2], kernel_size=(4, 4),     ##channels: 8 16 32 64
                                          stride=2, padding=1)
        self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                         out_channels=channels[2], stride=1,
                                         in_channels=channels[2])
        self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2])

        self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2] ,out_channels=channels[1], kernel_size=(4, 4),
                                          stride=2, padding=1)
        self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                         out_channels=channels[1], stride=1,
                                         in_channels=channels[1])
        self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1])

        self.head = _FCNHead(in_channels=channels[1], channels=classes, momentum=self.momentum)

        self.output_0 = nn.Conv2d(32, 1, 1)
        self.final = nn.Conv2d(4, 1, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)


    def _make_layer(self, block, out_channels, in_channels, blocks, stride):

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels , stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.groups, norm_layer=norm_layer))
        self.inplanes = out_channels  * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, self.groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _fuse_layer(self, fuse_mode, channels):

        if fuse_mode == 'AsymBi':
          fuse_layer = AsymBiChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def forward(self, x, warm_flag=True):

        _, _, hei, wid = x.shape

        x = self.stem(x)  # (4,16,120,120)
        c1 = self.layer1(x)  # (4,16,120,120)
        c2 = self.layer2(c1)  # (4,32, 60, 60)
        c3 = self.layer3(c2)  # (4,64, 30, 30)

        # print('c3:', c3.shape)
        deconvc2 = self.deconv2(c3)  # (4,32, 60, 60)
        fusec2 = self.fuse2(deconvc2, c2)  # (4,32, 60, 60)
        upc2 = self.uplayer2(fusec2)  # (4,32, 60, 60)
        # print('upc2:', upc2.shape)

        deconvc1 = self.deconv1(upc2)  # (4,16,120,120)
        fusec1 = self.fuse1(deconvc1, c1)  # (4,16,120,120)
        upc1 = self.uplayer1(fusec1)  # (4,16,120,120)
        # print('upc1:', upc1.shape)
        pred = self.head(upc1)  # (4,1,120,120)
        # print('pred:', pred.shape)
        out1 = F.interpolate(pred, scale_factor=2, mode='bilinear')  # down 4             # (4,1,480,480)
        # print('out1:', out1.shape)
        out = F.interpolate(out1, scale_factor=2, mode='bilinear')

        if warm_flag:
            mask0 = out
            mask1 = out1
            mask2 = pred
            mask3 = self.output_0(upc2)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            # print(f"mask0: {mask0.shape}")
            # print(f"mask1: {mask1.shape}")
            # print(f"mask2: {mask2.shape}")
            # print(f"mask3: {mask3.shape}")
            # print(f"output: {output.shape}")

            return [mask0, mask1, mask2, mask3], output
        else:
            output = out
            return [], output

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)



class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=inter_channels, out_channels=channels,kernel_size=1)
        )
    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.block(x)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# if __name__ == '__main__':
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
#     layers = [3] * 3
#     channels = [x * 1 for x in [8, 16, 32, 64]]
#     in_channels = 3
#     model = ASKCResUNet(in_channels, layers=layers, channels=channels, fuse_mode='AsymBi', classes=1)
#
#     model = model.cuda()
#     DATA = torch.randn(8,3,480,480).to(DEVICE)
#
#     output=model(DATA)
#     print("output:",np.shape(output))
if __name__ == '__main__':
    net = ASKCResUNet()
    x = torch.rand(8, 3, 256, 256)
    y = net(x, warm_flag=True)
    print()

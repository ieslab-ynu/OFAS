""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""


import torch.nn as nn
import math
import torch
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class OFASI(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, in_channel,out_channel):
        super(OFASI, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channel, in_channel*2)
        self.proj_1 = nn.Linear(in_channel*2, out_channel)
        self.prob_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.detach()
        x =self.avgpool(x).squeeze()
        proj = self.proj(x)
        proj_1= self.proj_1(proj).view(proj.size(0),-1,1)
        prob_Sigmoid = self.prob_Sigmoid(proj_1)
        disc_prob_skip=(prob_Sigmoid >= 0.5).float().detach() - prob_Sigmoid.detach() + prob_Sigmoid
        return disc_prob_skip

class OFASII(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, in_channel,out_channel):
        super(OFASII, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channel, in_channel*2)
        self.proj_1 = nn.Linear(in_channel*2, in_channel*2)
        self.proj_2 = nn.Linear(in_channel*2, out_channel)
        self.prob_Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.detach()
        x =self.avgpool(x).squeeze()
        proj = self.proj(x)




        proj_1 = self.proj_1(proj)
        proj_2= self.proj_2(proj_1).view(proj.size(0),-1,1)
        prob_Sigmoid = self.prob_Sigmoid(proj_2)
        disc_prob_skip=(prob_Sigmoid >= 0.5).float().detach() - prob_Sigmoid.detach() + prob_Sigmoid
        return disc_prob_skip


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
class GateResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, OFAS, blocks_num, layer_num=50, num_classes=10):
        self.inplanes = 16
        self.blocks_num = blocks_num
        super(GateResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.gate_layer = OFAS(in_channel=self.inplanes, out_channel=layer_num)
        self.layer1, self.ds1= self._make_layer(block, 16, blocks_num[0])
        self.layer2, self.ds2 = self._make_layer(block, 32, blocks_num[1], stride=2)
        self.layer3, self.ds3 = self._make_layer(block, 64, blocks_num[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), downsample

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        prev = x = self.relu(x)
        masks_skip = self.gate_layer(x)
        masks = []
        num = 0
        for i in range(0, self.blocks_num[0]):
            mask_skip = masks_skip[:, num].view(x.size(0), 1, 1, 1)
            x = self.layer1[i](x)
            prev = x = mask_skip.expand_as(x) * x + (1 - mask_skip).expand_as(prev) * prev
            num = num + 1
            masks.append(mask_skip.squeeze())
        for i in range(0, self.blocks_num[1]):
            mask_skip = masks_skip[:, num].view(x.size(0), 1, 1, 1)
            if i==0:
                prev = self.ds2(prev)
            x = self.layer2[i](x)
            prev = x = mask_skip.expand_as(x) * x + (1 - mask_skip).expand_as(prev) * prev
            num = num + 1
            masks.append(mask_skip.squeeze())
        for i in range(0, self.blocks_num[2]):
            mask_skip = masks_skip[:, num].view(x.size(0), 1, 1, 1)
            if i==0:
                prev = self.ds3(prev)
            x = self.layer3[i](x)
            prev = x = mask_skip.expand_as(x) * x + (1 - mask_skip).expand_as(prev) * prev
            num = num + 1
            masks.append(mask_skip.squeeze())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, masks

def cifar10_ofas_38(num_classes=10):
    """SkipNet-38 with FFGate-I"""
    return GateResNet(BasicBlock,OFASI, [6, 6, 6], layer_num=18, num_classes=num_classes)

def cifar100_ofas_38(num_classes=100):
    """SkipNet-38 with FFGate-I"""
    return GateResNet(BasicBlock,OFASI, [6, 6, 6], layer_num=18, num_classes=num_classes)

def cifar100_ofas_110(num_classes=100):
    """SkipNet-38 with FFGate-I"""
    return GateResNet(BasicBlock,OFASII, [18, 18, 18], layer_num=54, num_classes=num_classes)

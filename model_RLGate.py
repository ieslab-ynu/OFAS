""" This file contains the model definitions for both original ResNet (6n+2
layers) and SkipNets.
"""

from torch.distributions import Categorical
import torch.nn as nn
import math
import torch
from threading import Lock
global_lock = Lock()
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Rl_OFASI(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, in_channel,out_channel):
        super(Rl_OFASI, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channel, in_channel*2)
        self.proj_1 = nn.Linear(in_channel*2, out_channel)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = x.detach()
        x =self.avgpool(x).squeeze()
        proj = self.proj(x)
        proj_1= self.proj_1(proj).view(proj.size(0),-1,1)
        prob = self.prob(proj_1)
        prob_skip = torch.cat([prob,1-prob],dim=2)
        if self.training:
            skip=(prob_skip[:,:, 1] > 0.5).float().detach()*0.9-prob_skip[:,:, 0].detach()+prob_skip[:,:, 0]+(prob_skip[:,:, 0] >= 0.5).float().detach()*0.1
            execute=(prob_skip[:,:, 1] > 0.5).float().detach()*0.1-prob_skip[:,:, 0].detach()+prob_skip[:,:, 0]+(prob_skip[:,:, 0] >= 0.5).float().detach()*0.9
            bi_prob = torch.stack([skip, execute],2)
            dist = Categorical(bi_prob)
            action = dist.sample()
        else:
            dist = None
            action = (prob_skip[:,:, 0] >= 0.5).float()
        action_reshape = action.view(action.size(0), -1, 1, 1)

        return action_reshape, action, dist

class Rl_OFASII(nn.Module):
    """ Use Max Pooling First and then apply to multiple 2 conv layers.
    The first conv has stride = 1 and second has stride = 2"""
    def __init__(self, in_channel,out_channel):
        super(Rl_OFASII, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channel, in_channel*3)
        self.proj_1 = nn.Linear(in_channel*3, in_channel*2)
        self.proj_2 = nn.Linear(in_channel*2, out_channel)
        self.prob = nn.Sigmoid()

    def forward(self, x):
        x = x.detach()
        x =self.avgpool(x).squeeze()
        proj = self.proj(x)
        proj_1 = self.proj_1(proj)
        proj_2= self.proj_2(proj_1)
        prob = self.prob(proj_2).view(proj.size(0),-1,1)
        prob_skip = torch.cat([prob,1-prob],dim=2)
        if self.training:
            skip=(prob_skip[:,:, 1] > 0.5).float().detach()*0.9-prob_skip[:,:, 0].detach()+prob_skip[:,:, 0]+(prob_skip[:,:, 0] >= 0.5).float().detach()*0.1
            execute=(prob_skip[:,:, 1] > 0.5).float().detach()*0.1-prob_skip[:,:, 0].detach()+prob_skip[:,:, 0]+(prob_skip[:,:, 0] >= 0.5).float().detach()*0.9
            bi_prob = torch.stack([skip, execute],2)
            dist = Categorical(bi_prob)
            action = dist.sample()
        else:
            dist = None
            action = (prob_skip[:,:, 0] >= 0.5).float()
        action_reshape = action.view(action.size(0), -1, 1, 1)

        return action_reshape, action, dist
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
class RLGateResNet(nn.Module):
    """Original ResNet without routing modules"""
    def __init__(self, block, Rl_OFAS,  blocks_num, layer_num=50, num_classes=10):
        self.inplanes = 16
        self.blocks_num = blocks_num
        super(RLGateResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.gate_layer = Rl_OFAS(in_channel=self.inplanes, out_channel=layer_num)
        self.layer1, self.ds1= self._make_layer(block, 16, blocks_num[0])
        self.layer2, self.ds2 = self._make_layer(block, 32, blocks_num[1], stride=2)
        self.layer3, self.ds3 = self._make_layer(block, 64, blocks_num[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.saved_actions = {}
        self.saved_assactions = {}
        self.saved_outputs = {}
        self.saved_dists = {}
        self.saved_targets = {}
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

    def forward(self, x,target_var):
        x = self.conv1(x)
        x = self.bn1(x)
        prev = x = self.relu(x)
        masks_skip,actions_skip,dist = self.gate_layer(x)
        masks = []
        current_device = torch.cuda.current_device()
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
        with global_lock:
            self.saved_assactions[current_device] = actions_skip
            self.saved_actions[current_device] = masks
            self.saved_outputs[current_device] = x
            self.saved_targets[current_device] = target_var
            self.saved_dists[current_device] = dist
        return x, masks
def cifar10_rl_ofas_38(num_classes=10):
    """SkipNet-38 with FFGate-I"""
    return RLGateResNet(BasicBlock, Rl_OFASI, [6, 6, 6], layer_num=18, num_classes=num_classes)



def cifar100_rl_ofas_38(num_classes=100):
    """SkipNet-38 with FFGate-I"""
    return RLGateResNet(BasicBlock, Rl_OFASI, [6, 6, 6], layer_num=18, num_classes=num_classes)


def cifar100_rl_ofas_110(num_classes=100):
    """SkipNet-38 with FFGate-I"""
    return RLGateResNet(BasicBlock, Rl_OFASII, [18, 18, 18], layer_num=54, num_classes=num_classes)
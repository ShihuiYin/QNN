'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant import QuantizeConv2d, QuantizeLinear, QuantizeActLayer

class BasicBlock_quant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, w_bits=1, w_H=1., a_bits=1, a_H=1.):
        super(BasicBlock_quant, self).__init__()
        self.conv1 = QuantizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                n_bits=w_bits, H=w_H)
        self.bn1 = nn.BatchNorm2d(planes)
        self.quant_act = QuantizeActLayer(n_bits=a_bits, H=a_H)
        self.conv2 = QuantizeConv2d(planes, planes*self.expansion, kernel_size=3, stride=1, padding=1, bias=False,
                n_bits=w_bits, H=w_H)
        self.bn2 = nn.BatchNorm2d(planes*self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                QuantizeConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False,
                    n_bits=w_bits, H=w_H),
                #nn.BatchNorm2d(self.expansion*planes),
                #QuantizeActLayer(n_bits=a_bits, H=a_H)
            )

    def forward(self, x):
        out = self.quant_act(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.quant_act(self.bn2(out))
        return out


class Bottleneck_quant(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_quant, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_quant(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, w_bits=1, w_H=1., a_bits=1, a_H=1.):
        super(ResNet_quant, self).__init__()
        self.in_planes = 64

        self.conv1 = QuantizeConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False,
                n_bits=w_bits, H=w_H)
        self.bn1 = nn.BatchNorm2d(64)
        self.quant_act = QuantizeActLayer(n_bits=a_bits, H=a_H)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, 
                w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)
        self.linear = QuantizeLinear(512*block.expansion, num_classes, n_bits=w_bits, H=w_H)

    def _make_layer(self, block, planes, num_blocks, stride, w_bits=1, w_H=1., a_bits=1, a_H=1.):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, w_bits, w_H, a_bits, a_H))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.quant_act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18_quant(w_bits=1, w_H=1., a_bits=1, a_H=1.):
    return ResNet_quant(BasicBlock_quant, [2,2,2,2], w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)

def ResNet34_quant(w_bits=1, w_H=1., a_bits=1, a_H=1.):
    return ResNet_quant(BasicBlock_quant, [3,4,6,3], w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)

def ResNet50_quant(w_bits=1, w_H=1., a_bits=1, a_H=1.):
    return ResNet_quant(Bottleneck_quant, [3,4,6,3], w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)

def ResNet101_quant(w_bits=1, w_H=1., a_bits=1, a_H=1.):
    return ResNet_quant(Bottleneck_quant, [3,4,23,3], w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)

def ResNet152_quant(w_bits=1, w_H=1., a_bits=1, a_H=1.):
    return ResNet_quant(Bottleneck_quant, [3,8,36,3], w_bits=w_bits, w_H=w_H, a_bits=a_bits, a_H=a_H)


def test():
    net = ResNet18_quant()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()

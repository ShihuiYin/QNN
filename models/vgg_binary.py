'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from quant import BinarizeConv2d, BinarizeLinear, BinarizeActLayer
BinarizeConv2d = nn.Conv2d
BinarizeLinear = nn.Linear
#BinarizeActLayer = nn.Hardtanh

cfg = {
    'VGG': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_binary(nn.Module):
    def __init__(self, vgg_name, fc=1024):
        super(VGG_binary, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        num_maxpooling_layers = cfg[vgg_name].count('M')
        last_conv_layer_output_dim = 512 * (4 ** (5 - num_maxpooling_layers))
        self.classifier = nn.Sequential(
                BinarizeLinear(last_conv_layer_output_dim, fc),
                nn.BatchNorm1d(fc),
                BinarizeActLayer(),
                BinarizeLinear(fc, fc),
                nn.BatchNorm1d(fc),
                BinarizeActLayer(),
                BinarizeLinear(fc, 10)
                )
        # self.regime = {
        #         0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-2},
        #         40: {'lr': 1e-3},
        #         80: {'lr': 5e-4},
        #         100: {'lr': 1e-4},
        #         120: {'lr': 5e-5},
        #         140: {'lr': 1e-5}
        #         }

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if in_channels == 3:
                layers += [BinarizeConv2d(in_channels, x, kernel_size=3, padding=1)]
                layers += [nn.BatchNorm2d(x)]
                in_channels = x
            else:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [BinarizeActLayer()]
                    layers += [BinarizeConv2d(in_channels, x, kernel_size=3, padding=1),
                               nn.BatchNorm2d(x)]
                    in_channels = x
        layers += [BinarizeActLayer()]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

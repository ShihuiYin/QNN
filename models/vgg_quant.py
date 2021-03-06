'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from .quant import QuantizeConv2d, QuantizeLinear, QuantizeActLayer

cfg = {
    'VGG': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGGS': [128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
    'VGGT': [128, 128, 'M', 128, 256, 'M', 256, 256, 'M'],
    'VGGD': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_quant(nn.Module):
    def __init__(self, vgg_name, a_bits=2, w_bits=2, fc=1024, a_H=1., w_H=1.):
        super(VGG_quant, self).__init__()
        self.a_bits = a_bits
        self.w_bits = w_bits
        self.a_H = a_H
        self.w_H = w_H
        self.features = self._make_layers(cfg[vgg_name])
        num_maxpooling_layers = cfg[vgg_name].count('M')
        if 'S' in vgg_name or 'T' in vgg_name:
            last_conv_layer_output_dim = 256 * (4 ** (5 - num_maxpooling_layers))
        elif 'D' in vgg_name:
            last_conv_layer_output_dim = 512 * (4 ** (5 - num_maxpooling_layers))
        else:
            last_conv_layer_output_dim = 512 * (4 ** (5 - num_maxpooling_layers))
        self.classifier = nn.Sequential(
                QuantizeLinear(last_conv_layer_output_dim, fc, n_bits=w_bits, H=w_H),
                nn.BatchNorm1d(fc),
                QuantizeActLayer(n_bits=a_bits, H=a_H),
                QuantizeLinear(fc, fc, n_bits=w_bits, H=w_H),
                nn.BatchNorm1d(fc),
                QuantizeActLayer(n_bits=a_bits, H=a_H),
                QuantizeLinear(fc, 10, n_bits=w_bits, H=w_H),
                )
       #self.regime = {
       #    0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
       #    40: {'lr': 1e-3},
       #    80: {'lr': 5e-4},
       #    100: {'lr': 1e-4},
       #    120: {'lr': 5e-5},
       #    140: {'lr': 1e-5}
       #}
        
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
                layers += [QuantizeConv2d(in_channels, x, kernel_size=3, padding=1, n_bits=self.w_bits, H=self.w_H)]
                layers += [nn.BatchNorm2d(x)]
                in_channels = x
            else:
                if x == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [QuantizeActLayer(n_bits=self.a_bits, H=self.a_H)]
                    layers += [QuantizeConv2d(in_channels, x, kernel_size=3, padding=1, n_bits=self.w_bits, H=self.w_H),
                               nn.BatchNorm2d(x)]
                    in_channels = x
        layers += [QuantizeActLayer(n_bits=self.a_bits, H=self.a_H)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()

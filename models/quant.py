from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def Quantize(tensor, H=1., n_bits=2):
    if n_bits == 1:
        return tensor.sign() * H
    if isinstance(H, torch.Tensor):
        tensor=torch.round((torch.clamp(tensor, -H.data, H.data)+H.data) * (2**n_bits - 1) / (2*H.data)) * 2*H.data / (2**n_bits - 1) - H.data
    else:
        tensor=torch.round((torch.clamp(tensor, -H, H)+H) * (2**n_bits - 1) / (2*H)) * 2*H / (2**n_bits - 1) - H
    return tensor

def Quant_k(x, H=64., levels=11, mode='software'): # levels should be an odd number
    scale = (levels - 1) / (2*H)
    y = nn.functional.hardtanh(x, -H, H)
    y.data.mul_(scale).round_()
    if mode == 'software':
        scale_inv = 1. / scale
        y.data.mul_(scale_inv)
    return y

class QuantizeActLayer(nn.Module):
    def __init__(self, n_bits=2, H=1., inplace=True):
        super(QuantizeActLayer, self).__init__()
        self.inplace = inplace
        self.n_bits = n_bits

    def forward(self, x):
        y = nn.functional.hardtanh(x)
        with torch.no_grad():
            y.data = Quantize(y.data, n_bits=self.n_bits)
        return y

    def extra_repr(self):
        return super(QuantizeActLayer, self).extra_repr() + 'n_bits={}'.format(self.n_bits)

class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs.pop('n_bits')
        self.H_init = kwargs.pop('H')
        self.sram_depth = 0 # will be over-written in main.py
        self.quant_bound = 64
        self.mode = 'software'
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, n_bits=self.n_bits)
        if self.sram_depth > 0:
            self.weight_list = torch.split(self.weight, self.sram_depth, dim=1)
            input_list = torch.split(input, self.sram_depth, dim=1)
            out = 0
            for input_p, weight_p in zip(input_list, self.weight_list):
                partial_sum = nn.functional.linear(input_p, weight_p)
                partial_sum_quantized = Quant_k(partial_sum, self.quant_bound, mode=self.mode)
                out += partial_sum_quantized
        else:
            out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self):
        return super(QuantizeLinear, self).extra_repr() + ', n_bits={}'.format(self.n_bits)

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs.pop('n_bits')
        self.H_init = kwargs.pop('H')
        self.sram_depth = 0 # will be over-written in main.py
        self.quant_bound = 64
        self.mode = 'software'
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, n_bits=self.n_bits)
        if self.sram_depth > 0 and input.shape[1] > 3:
            input_padded = torch.nn.functional.pad(input, [self.padding[0]]*4)
            input_list = torch.split(input_padded, self.sram_depth, dim=1)
            self.weight_list = torch.split(self.weight, self.sram_depth, dim=1)
            out = 0
            map_x, map_y = input.shape[2], input.shape[3]
            for input_p, weight_p in zip(input_list, self.weight_list):
                for k in range(self.weight.shape[2]):
                    for j in range(self.weight.shape[3]):
                        input_kj = input_p[:,:,k:k+map_x, j:j+map_y] #.contiguous()
                        weight_kj = weight_p[:,:,k:k+1,j:j+1] #.contiguous()
                        partial_sum = nn.functional.conv2d(input_kj, weight_kj, None, self.stride, (0,0), self.dilation, self.groups)
                        partial_sum_quantized = Quant_k(partial_sum, self.quant_bound, mode=self.mode)
                        out += partial_sum_quantized

        else:
            out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                       self.padding, self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

    def extra_repr(self):
        return super(QuantizeConv2d, self).extra_repr() + ', n_bits={}'.format(self.n_bits) 

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_status=True):
        self.mode = 'software' # 'software': normal batchnorm in software; 'hardware': simplified batchnorm in hardware
        self.preceding_bias = torch.zeros((num_features,)).cuda()
        self.preceding_scale = 1.
        self.weight_effective = torch.ones((num_features,)).cuda()
        self.bias_effective = torch.zeros((num_features,)).cuda()
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_status)

    def forward(self, input):
        if self.mode == 'software':
            return super(BatchNorm2d, self).forward(input)
        else:
            return input * self.weight_effective.view(1, -1, 1, 1).expand_as(input) + self.bias_effective.view(1, -1, 1, 1).expand_as(input)

class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_status=True):
        self.mode = 'software' # 'software': normal batchnorm in software; 'hardware': simplified batchnorm in hardware
        self.preceding_bias = torch.zeros((num_features,)).cuda()
        self.preceding_scale = 1.
        self.weight_effective = torch.ones((num_features,)).cuda()
        self.bias_effective = torch.zeros((num_features,)).cuda()
        super(BatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_status)

    def forward(self, input):
        if self.mode == 'software':
            return super(BatchNorm1d, self).forward(input)
        else:
            return input * self.weight_effective.view(1, -1).expand_as(input) + self.bias_effective.view(1, -1).expand_as(input)



# Copyright 2020   Shihui Yin    Arizona State University

# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# Description: functions for weight/activation quantization and input splitting
# Created on 02/09/2020
import torch
import torch.nn as nn
from torch.autograd import Function
import math


def Binarize(tensor):
    return tensor.sign()

def Quantize(tensor, n_bits=2):
    #tensor.clamp_(-1., 1.)
    if n_bits == 1:
        return tensor.sign()
    tensor=torch.round((tensor+1.) * (2**n_bits - 1) / 2.) * 2 / (2**n_bits - 1) - 1.
    return tensor

class BinarizeAct(Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class QuantizeAct(Function):
    @staticmethod
    def forward(ctx, input, n_bits=2):
        return Quantize(input, n_bits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class BinarizeActLayer(nn.Module):
    def __init__(self, inplace=True):
        super(BinarizeActLayer, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return BinarizeAct.apply(nn.functional.hardtanh(x, inplace=self.inplace))

class QuantizeActLayer(nn.Module):
    def __init__(self, n_bits=2, inplace=True):
        super(QuantizeActLayer, self).__init__()
        self.inplace = inplace
        self.n_bits = n_bits

    def forward(self, x):
        return QuantizeAct.apply(nn.functional.hardtanh(x, inplace=self.inplace), self.n_bits)

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
            with torch.no_grad():
                self.weight.std = torch.std(self.weight.data)
        self.weight.data=Binarize(self.weight.org) * self.weight.std
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        kwargs.pop('n_bits')
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, self.n_bits)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
            with torch.no_grad():
                self.weight.std = torch.std(self.weight.data)
        self.weight.data=Binarize(self.weight.org) * self.weight.std

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        kwargs.pop('n_bits')
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, self.n_bits)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

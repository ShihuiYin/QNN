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


def Quantize(tensor, n_bits=2):
    if n_bits == 1:
        return tensor.sign()
    tensor=torch.round((torch.clamp(tensor, -1., 1.)+1.) * (2**n_bits - 1) / 2.) * 2 / (2**n_bits - 1) - 1.
    return tensor

class Quantize_STE_clipped(Function):
    @staticmethod
    def forward(ctx, input, n_bits=2):
        ctx.save_for_backward(input)
        return Quantize(input, n_bits)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) > 1] = 0
        return grad_input, None

class Quantize_STE_identity(Function):
    '''
    Currently, it's basically the same with QuantizeAct function
    '''
    @staticmethod
    def forward(ctx, input, n_bits=2):
        return Quantize(input, n_bits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

QuantizeAct = Quantize_STE_clipped
QuantizeWeight = Quantize_STE_identity
    
class QuantizeActLayer(nn.Module):
    def __init__(self, n_bits=2, inplace=True):
        super(QuantizeActLayer, self).__init__()
        self.inplace = inplace
        self.n_bits = n_bits

    def forward(self, x):
        return QuantizeAct.apply(nn.functional.hardtanh(x, inplace=self.inplace), self.n_bits)

    def extra_repr(self):
        return super(QuantizeActLayer, self).extra_repr() + 'n_bits={}'.format(self.n_bits)

class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        kwargs.pop('n_bits')
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        out = nn.functional.linear(input, QuantizeWeight.apply(self.weight, self.n_bits))
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self):
        return super(QuantizeLinear, self).extra_repr() + ', n_bits={}'.format(self.n_bits)

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        kwargs.pop('n_bits')
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        out = nn.functional.conv2d(input, QuantizeWeight.apply(self.weight, self.n_bits), 
                None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

    def extra_repr(self):
        return super(QuantizeConv2d, self).extra_repr() + ', n_bits={}'.format(self.n_bits)

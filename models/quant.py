import torch
import torch.nn as nn
from torch.autograd import Function

def Quantize(tensor, n_bits=2, H=1.):
    if n_bits == 1:
        return tensor.sign() * H
    tensor=torch.round((torch.clamp(tensor, -H, H)+H) * (2**n_bits - 1) / (2*H)) * 2*H / (2**n_bits - 1) - H
    return tensor

class Quantize_STE_clipped(Function):
    @staticmethod
    def forward(ctx, input, n_bits=2, H=1.):
        ctx.save_for_backward(input)
        ctx.H = H
        return Quantize(input, n_bits, H)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) > ctx.H] = 0
        return grad_input, None, None

class Quantize_STE_identity(Function):
    @staticmethod
    def forward(ctx, input, n_bits=2, H=1.):
        return Quantize(input, n_bits, H)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

QuantizeAct = Quantize_STE_clipped
QuantizeWeight = Quantize_STE_clipped
    
class QuantizeActLayer(nn.Module):
    def __init__(self, n_bits=2, H=1., inplace=True):
        super(QuantizeActLayer, self).__init__()
        self.inplace = inplace
        self.n_bits = n_bits
        self.H = H

    def forward(self, x):
        return QuantizeAct.apply(nn.functional.hardtanh(x, inplace=self.inplace), self.n_bits, self.H)

    def extra_repr(self):
        return super(QuantizeActLayer, self).extra_repr() + 'n_bits={}'.format(self.n_bits) + ', H={}'.format(self.H)

class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        self.H = kwargs['H']
        kwargs.pop('n_bits')
        kwargs.pop('H')
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        out = nn.functional.linear(input, QuantizeWeight.apply(self.weight, self.n_bits, self.H))
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self):
        return super(QuantizeLinear, self).extra_repr() + ', n_bits={}'.format(self.n_bits) + ', H={}'.format(self.H)

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        self.H = kwargs['H']
        kwargs.pop('n_bits')
        kwargs.pop('H')
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        out = nn.functional.conv2d(input, QuantizeWeight.apply(self.weight, self.n_bits, self.H), 
                None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

    def extra_repr(self):
        return super(QuantizeConv2d, self).extra_repr() + ', n_bits={}'.format(self.n_bits) + ', H={}'.format(self.H)

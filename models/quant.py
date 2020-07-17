import torch
import torch.nn as nn
from torch.autograd import Function

def Quantize(tensor, H=1., n_bits=2):
    if n_bits == 1:
        return tensor.sign() * H
    if isinstance(H, torch.Tensor):
        tensor=torch.round((torch.clamp(tensor, -H.data, H.data)+H.data) * (2**n_bits - 1) / (2*H.data)) * 2*H.data / (2**n_bits - 1) - H.data
    else:
        tensor=torch.round((torch.clamp(tensor, -H, H)+H) * (2**n_bits - 1) / (2*H)) * 2*H / (2**n_bits - 1) - H
    return tensor

class Quantize_STE_clipped(Function):
    @staticmethod
    def forward(ctx, input, n_bits):
        ctx.save_for_backward(input)
        ctx.n_bits = n_bits
        return Quantize(input, n_bits=n_bits)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) > 1.] = 0
        return grad_input, None

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
        self.n_bits = kwargs['n_bits']
        self.H_init = kwargs['H']
        kwargs.pop('n_bits')
        kwargs.pop('H')
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        #out = nn.functional.linear(input, Quantize_STE_clipped.apply(self.weight, self.n_bits))
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, n_bits=self.n_bits)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self):
        return super(QuantizeLinear, self).extra_repr() + ', n_bits={}'.format(self.n_bits)

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        self.H_init = kwargs['H']
        kwargs.pop('n_bits')
        kwargs.pop('H')
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        #out = nn.functional.conv2d(input, Quantize_STE_clipped.apply(self.weight, self.n_bits), 
        #        None, self.stride, self.padding, self.dilation, self.groups)

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Quantize(self.weight.org, n_bits=self.n_bits)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

    def extra_repr(self):
        return super(QuantizeConv2d, self).extra_repr() + ', n_bits={}'.format(self.n_bits)

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

def Quantize2(n_bits):
    if n_bits == 1:
        def binarize(x, H):
            return x.sign() * H
        return binarize
    k = 2**n_bits - 1.
    def quantize(x, H):
        with torch.no_grad():
            xq = (torch.round((torch.clamp(x/(2*H), -0.5, 0.5) + 0.5) * k) / k - 0.5) * 2 * H
        return xq
    return quantize

class Quantize_STE_clipped(Function):
    @staticmethod
    def forward(ctx, input, H, n_bits):
        ctx.save_for_backward(input, H)
        ctx.n_bits = n_bits
        return Quantize(input, H, n_bits)

    @staticmethod
    def backward(ctx, grad_output):
        input, H, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_H = torch.sum(Quantize(input/H, 1, ctx.n_bits) * grad_output.clone()).clamp_(-0.001, 0.001)
        grad_input[abs(input) > H] = 0
        return grad_input, grad_H, None

def Quantize_STE(n_bits):
    quant = Quantize2(n_bits)
    class Quantize_STE_clipped2(Function):
        @staticmethod
        def forward(ctx, input, H):
            output = quant(input, H)
            ctx.save_for_backward(input, H, output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, H, output = ctx.saved_tensors
            grad_input = grad_output.clone()
            #grad_H = torch.sum(quant(input/H, 1) * grad_output.clone()).clamp_(-0.003, 0.003)
            #grad_H = torch.sum(output/H*(output-input)).clamp_(-0.001, 0.001)
            # add regularization to H to minimize quantization error
            #grad_H = torch.sum(output/H*(1e-4*(output-input)+grad_output)).clamp_(-0.001, 0.001)
            grad_H = torch.sum(output/H*grad_output).clamp_(-0.001, 0.001)
            grad_input[abs(input) > H] = 0
            return grad_input, grad_H
    return Quantize_STE_clipped2

class Quantize_STE_identity(Function):
    @staticmethod
    def forward(ctx, input, H, n_bits):
        return Quantize(input, H, n_bits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_H = torch.sum(torch.sign(input) * grad_output.clone())
        return grad_input, grad_H, None

#QuantizeAct = Quantize_STE_clipped2
#QuantizeWeight = Quantize_STE_clipped2
    
class QuantizeActLayer(nn.Module):
    def __init__(self, n_bits=2, H=1., inplace=True):
        super(QuantizeActLayer, self).__init__()
        self.inplace = inplace
        self.n_bits = n_bits
        self.H_init = H
        self.H = nn.Parameter(data=torch.Tensor(1),requires_grad=True)
        self.H.data = torch.tensor(self.H_init)
        self.quantize = Quantize_STE(n_bits)

    def forward(self, x):
        return self.quantize.apply(x, self.H)

    def extra_repr(self):
        return super(QuantizeActLayer, self).extra_repr() + 'n_bits={}'.format(self.n_bits) + ', H={}'.format(self.H)

class QuantizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        self.H_init = kwargs['H']
        kwargs.pop('n_bits')
        kwargs.pop('H')
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.H = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        #self.H.data = torch.tensor(self.H_init)
        self.H.data = 2 * torch.std(self.weight.data)
        self.quantize = Quantize_STE(self.n_bits).apply

    def forward(self, input):
        out = nn.functional.linear(input, self.quantize(self.weight, self.H))
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out

    def extra_repr(self):
        return super(QuantizeLinear, self).extra_repr() + ', n_bits={}'.format(self.n_bits) + ', H={}'.format(self.H)

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        self.n_bits = kwargs['n_bits']
        self.H_init = kwargs['H']
        kwargs.pop('n_bits')
        kwargs.pop('H')
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)
        self.H = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        #self.H.data = torch.tensor(self.H_init)
        self.H.data = 2 * torch.std(self.weight.data)
        self.quantize = Quantize_STE(self.n_bits).apply

    def forward(self, input):
        out = nn.functional.conv2d(input, self.quantize(self.weight, self.H), 
                None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

    def extra_repr(self):
        return super(QuantizeConv2d, self).extra_repr() + ', n_bits={}'.format(self.n_bits) + ', H={}'.format(self.H)

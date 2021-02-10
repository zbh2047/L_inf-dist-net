import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import norm_dist_cpp

class DistF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group, p, tag):
        output = torch.empty(x.size(0), weight.size(0), x.size(2),
                             device=x.device, requires_grad=x.requires_grad or weight.requires_grad)
        assert weight.size(1) * group == x.size(1)
        if math.isinf(p):
            pos = torch.empty(x.size(0), weight.size(0), x.size(2), dtype=torch.int,
                              device=x.device, requires_grad=False)
            norm_dist_cpp.inf_dist_forward(x, weight, output, pos, group)
            ctx.save_for_backward(x, weight, output, pos)
        elif p > 1:
            norm_dist_cpp.norm_dist_forward(x, weight, output, group, p)
            ctx.save_for_backward(x, weight, output)
        else:
            raise NotImplementedError
        ctx.group = group
        ctx.p = p
        ctx.tag = tag
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = None
        grad_output = grad_output.contiguous()
        if math.isinf(ctx.p):
            x, weight, output, pos = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(x, device=grad_output.device) # it must be set to zero
                norm_dist_cpp.inf_dist_backward_input(grad_output, pos, grad_input, ctx.group)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight, device=grad_output.device)  # it must be set to zero
                norm_dist_cpp.inf_dist_backward_weight(grad_output, pos, grad_weight, ctx.group)
        else:
            x, weight, output = ctx.saved_tensors
            if ctx.needs_input_grad[0]:
                grad_input = torch.empty_like(x, device=grad_output.device)
                norm_dist_cpp.norm_dist_backward_input(grad_output, x, weight, output, grad_input, ctx.group, ctx.p)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight, device=grad_output.device) # it must be set to zero
                norm_dist_cpp.norm_dist_backward_weight(grad_output, x, weight, output, grad_weight, ctx.group, ctx.p)
        return grad_input, grad_weight, None, None, None

def norm_dist(input, weight, group, p, use_custom_cuda_func=True, tag=None):
    if use_custom_cuda_func:
        y = DistF.apply(input, weight, group, p, tag)
    else:
        y = input.view(input.size(0), group, 1, -1, input.size(2)) - weight.view(group, -1, weight.size(-1), 1)
        with torch.no_grad():
            normalize = torch.norm(y, dim=3, p=float('inf'), keepdim=True)
        y = torch.norm(y / normalize, dim=3, p=p, keepdim=True) * normalize
        y = y.view(y.size(0), -1, y.size(-1))
    return y

def inf_dist_bound(input_lower, input_upper, weight, group, use_custom_cuda_func=True, tag=None):
    shape = input_lower.size(0), weight.size(0), input_lower.size(2)
    if use_custom_cuda_func:
        output_lower = input_lower.new_empty(*shape, requires_grad=False)
        output_upper = input_upper.new_empty(*shape, requires_grad=False)
        norm_dist_cpp.inf_dist_bound_forward(input_lower, input_upper, weight, output_lower, output_upper, group)
    else:
        w = weight.view(group, -1, weight.size(-1), 1)
        y1 = input_lower.view(input_lower.size(0), group, 1, -1, input_lower.size(2)) - w
        y2 = input_upper.view(input_upper.size(0), group, 1, -1, input_upper.size(2)) - w
        abs_y1 = torch.abs(y1)
        abs_y2 = torch.abs(y2)
        output_upper = torch.maximum(abs_y1, abs_y2).max(dim=3)[0].view(shape)
        output_lower = torch.minimum(abs_y1, abs_y2)
        output_lower[(y1 < 0) & (y2 > 0)] = 0
        output_lower = output_lower.max(dim=3)[0].view(shape)
    return output_lower, output_upper

class MeanNorm(nn.Module):
    def __init__(self, out_channels, momentum=0.1):
        super(MeanNorm, self).__init__()
        self.out_channels = out_channels
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(out_channels))

    def forward(self, x):
        y = x.view(x.size(0), x.size(1), -1)
        if self.training:
            if x.dim() > 2:
                mean = y.mean(dim=-1).mean(dim=0)
            else:
                mean = x.mean(dim=0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        else:
            mean = self.running_mean
        x = (y - mean.unsqueeze(-1)).view_as(x)
        return x

    def extra_repr(self):
        return '{num_features}'.format(num_features=self.out_channels)

class NormDistConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, group=1, p=0,
                 bias=True, mean_normalize=False, identity_init=True):
        super(NormDistConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        self.p = p
        assert(in_channels % group == 0)
        assert (out_channels % group == 0)

        weight_tensor = torch.randn(out_channels, in_channels // group, kernel_size, kernel_size)
        if identity_init and in_channels <= out_channels:
            for i in range(out_channels):
                weight_tensor[i, i % (in_channels // group), kernel_size // 2, kernel_size // 2] = -10.0
        self.weight = nn.Parameter(weight_tensor)
        self.normalize = MeanNorm(out_channels) if mean_normalize else None
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        if not hasattr(NormDistConv, 'tag'):
            NormDistConv.tag = 0
        NormDistConv.tag += 1
        self.tag = NormDistConv.tag

    def forward(self, x, lower=None, upper=None):
        h, w = x.size(2), x.size(3)
        x = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        weight = self.weight.view(self.weight.size(0), -1)
        y = norm_dist(x.view(x.size(0), x.size(1), -1), weight, self.group, p=self.p, tag=self.tag)
        y = y.view(x.size(0), -1, (h + 2 * self.padding - self.kernel_size) // self.stride + 1,
                   (w + 2 * self.padding - self.kernel_size) // self.stride + 1)
        if self.normalize is not None:
            y = self.normalize(y)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        if lower is None or upper is None:
            return y, None, None
        assert math.isinf(self.p) and not self.training
        lower = F.unfold(lower, self.kernel_size, self.dilation, self.padding, self.stride)
        upper = F.unfold(upper, self.kernel_size, self.dilation, self.padding, self.stride)
        lower, upper = inf_dist_bound(lower.view(x.size(0), x.size(1), -1), upper.view(x.size(0), x.size(1), -1),
                                      weight, self.group, tag=self.tag)
        lower = lower.view_as(y)
        upper = upper.view_as(y)
        if self.normalize is not None:
            lower = self.normalize(lower)
            upper = self.normalize(upper)
        if self.bias is not None:
            lower = lower + self.bias.view(1, -1, 1, 1)
            upper = upper + self.bias.view(1, -1, 1, 1)
        return y, lower, upper

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.group != 1:
            s += ', group={group}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class NormDist(NormDistConv):
    def __init__(self, in_features, out_features, p=0, bias=True, mean_normalize=False, identity_init=True):
        super(NormDist, self).__init__(in_features, out_features, 1, p=p, bias=bias, mean_normalize=mean_normalize,
                                   identity_init=identity_init)
    def forward(self, x, lower=None, upper=None):
        x = x.unsqueeze(-1).unsqueeze(-1)
        lower = lower.view_as(x) if lower is not None else None
        upper = upper.view_as(x) if upper is not None else None
        x, lower, upper = super(NormDist, self).forward(x, lower, upper)
        x = x.squeeze(-1).squeeze(-1)
        lower = lower.view_as(x) if lower is not None else None
        upper = upper.view_as(x) if upper is not None else None
        return x, lower, upper
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

def set_p_norm(model, p):
    for m in model.modules():
        if isinstance(m, NormDist) or isinstance(m, NormDistConv):
            m.p = p

def get_p_norm(model):
    for m in model.modules():
        if isinstance(m, NormDist) or isinstance(m, NormDistConv):
            return m.p
    return None

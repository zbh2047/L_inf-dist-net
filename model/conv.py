import torch.nn as nn
from model.norm_dist import NormDistConv, NormDist
import numpy as np
net_width = 32

class Conv(nn.Module):
    def __init__(self, width, input_dim, hidden=512, num_classes=10):
        super(Conv, self).__init__()
        pixels = input_dim[1] * input_dim[2]
        width *= net_width
        conv = []
        conv.append(NormDistConv(input_dim[0], width, 3, bias=False, mean_normalize=True, padding=1))
        conv.append(NormDistConv(width, width, 3, bias=False, mean_normalize=True, padding=1))
        pixels //= 4
        width *= 2
        conv.append(NormDistConv(width // 2, width, 3, bias=False, padding=1, stride=2, mean_normalize=True))
        conv.append(NormDistConv(width, width, 3, bias=False, padding=1, mean_normalize=True))
        conv.append(NormDistConv(width, width, 3, bias=False, padding=1, mean_normalize=True))
        self.conv = nn.ModuleList(conv)
        self.fc = NormDist(pixels * width, hidden, bias=False, mean_normalize=True)
        self.fc_last = NormDist(hidden, num_classes, bias=True, mean_normalize=False)
    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.conv:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        paras = self.fc(*paras)
        paras = self.fc_last(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class ConvFeature(nn.Module):
    def __init__(self, width, input_dim, hidden=512):
        super(ConvFeature, self).__init__()
        pixels = input_dim[1] * input_dim[2]
        width *= net_width
        conv = []
        conv.append(NormDistConv(input_dim[0], width, 3, bias=False, mean_normalize=True, padding=1))
        conv.append(NormDistConv(width, width, 3, bias=False, mean_normalize=True, padding=1))
        pixels //= 4
        width *= 2
        conv.append(NormDistConv(width // 2, width, 3, bias=False, padding=1, stride=2, mean_normalize=True))
        conv.append(NormDistConv(width, width, 3, bias=False, padding=1, mean_normalize=True))
        conv.append(NormDistConv(width, width, 3, bias=False, padding=1, mean_normalize=True))
        self.conv = nn.ModuleList(conv)
        self.fc = NormDist(pixels * width, hidden, bias=False, mean_normalize=True)
        self.out_features = hidden
    def forward(self, x, lower=None, upper=None):
        paras = (x, lower, upper)
        for layer in self.conv:
            paras = layer(*paras)
        paras = [None if y is None else y.view(y.size(0), -1) for y in paras]
        paras = self.fc(*paras)
        return paras

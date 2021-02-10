import torch.nn as nn
import numpy as np
from model.norm_dist import NormDist

net_width = 1024

class MLP(nn.Module):
    def __init__(self, depth, width, input_dim, num_classes=10):
        super(MLP, self).__init__()
        layers = []
        input_dim = np.prod(input_dim)
        for i in range(depth - 1):
            layers.append(NormDist(input_dim, net_width * width, bias=False, mean_normalize=True))
            input_dim = net_width * width
        self.fc = nn.ModuleList(layers)
        self.fc_last = NormDist(input_dim, num_classes, bias=True, mean_normalize=False)
    def forward(self, x, lower=None, upper=None):
        paras = [None if y is None else y.view(y.size(0), -1) for y in (x, lower, upper)]
        for layer in self.fc:
            paras = layer(*paras)
        paras = self.fc_last(*paras)
        paras = [None if y is None else -y for y in (paras[0], paras[2], paras[1])]
        return paras

class MLPFeature(nn.Module):
    def __init__(self, depth, width, input_dim):
        super(MLPFeature, self).__init__()
        layers = []
        input_dim = np.prod(input_dim)
        for i in range(depth):
            layers.append(NormDist(input_dim, net_width * width, bias=False, mean_normalize=True))
            input_dim = net_width * width
        self.out_features = input_dim
        self.fc = nn.ModuleList(layers)
    def forward(self, x, lower=None, upper=None):
        paras = [None if y is None else y.view(y.size(0), -1) for y in (x, lower, upper)]
        for layer in self.fc:
            paras = layer(*paras)
        return paras

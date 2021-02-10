import torch.nn as nn

class Model(nn.Module):
    def __init__(self, feature, predictor, eps):
        super(Model, self).__init__()
        self.feature = feature
        self.predictor = predictor
        self.eps = eps
    def forward(self, x, lower=None, upper=None, targets=None):
        if targets is None:
            lower = upper = None
        if self.feature is not None:
            x, lower, upper = self.feature(x, lower=lower, upper=upper)
        if targets is not None and (lower is None or upper is None):
            lower = x - self.eps
            upper = x + self.eps
        return self.predictor(x, lower, upper, targets=targets)

def set_eps(model, eps):
    for m in model.modules():
        if isinstance(m, Model):
            m.eps = eps

def get_eps(model):
    for m in model.modules():
        if isinstance(m, Model):
            return m.eps
    return None
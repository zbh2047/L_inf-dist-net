import math
import torch
from torch.optim import Optimizer
from model.norm_dist import get_p_norm

class AdamW(Optimizer):
    '''
    Implements AdamW algorithm with \ell_p weight decay.
    Note: for each tensor, let its shape be [C_1, C_2, ..., C_d], then \ell_p weight decay are applied
        for each row which has shape [C_2, ..., C_d]. For bias, the weight decay is applied elementwise.
    '''

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = [{'params': p, 'module': m} for m in model.modules() for p in m.parameters(False)]
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p_norm = get_p_norm(group['module'])
                # x_i -= (|x_i|/\|x\|_p)^{p-2} x_i
                if p_norm is None:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                elif not math.isinf(p_norm):
                    sqr_p = p.view(p.size(0), -1)
                    sqr_p = sqr_p * sqr_p + group['eps']
                    sqr_p = sqr_p / sqr_p.max(dim=1, keepdim=True)[0]
                    pow_p = torch.pow(sqr_p, p_norm / 2 - 1)
                    sum_of_pow_p = torch.bmm(sqr_p.unsqueeze(1), pow_p.unsqueeze(2))
                    normalize_p = torch.pow(sum_of_pow_p, 2 / p_norm - 1).view(-1, 1) * pow_p
                    p.addcmul_(normalize_p.view_as(p), p, value=-group['lr'] * group['weight_decay'])
                else:
                    p2 = p.view(p.size(0), -1)
                    index = torch.max(p2.abs(), dim=1, keepdim=True)[1]
                    value = -group['lr'] * group['weight_decay'] * torch.gather(p2, index=index, dim=1)
                    p2.scatter_add_(dim=1, index=index, src=value)

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
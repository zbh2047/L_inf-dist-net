import torch
import torch.nn.functional as F

class AttackPGD():
    def __init__(self, net, eps, step_size, num_steps, up, down, random_start=True):
        super(AttackPGD, self).__init__()
        self.net = net
        self.rand = random_start
        self.step_size = step_size
        self.num_steps = num_steps
        self.eps = eps
        self.up = up
        self.down = down

    def find(self, inputs, targets):
        requires_grads = [x.requires_grad for x in self.net.parameters()]
        self.net.requires_grad_(False)

        x = inputs.detach()
        if self.rand:
            init_noise = torch.zeros_like(x).normal_(0, self.eps / 4)
            x = x + torch.clamp(init_noise, -self.eps / 2, self.eps / 2)
            x = torch.min(torch.max(x, self.down), self.up)

        for i in range(self.num_steps):
            x.requires_grad_()
            logits = self.net(x)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            loss.backward()
            x = torch.add(x.detach(),torch.sign(x.grad.detach()), alpha=self.step_size)
            x = torch.min(torch.max(x, inputs - self.eps), inputs + self.eps)
            x = torch.min(torch.max(x, self.down), self.up)

        for p, r in zip(self.net.parameters(), requires_grads):
            p.requires_grad_(r)
        return x

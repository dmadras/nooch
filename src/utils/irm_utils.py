import torch
from torch import autograd
from utils.utils import mean_nll, device

def irm_penalize(logits, y, no_cuda):
    scale = device(torch.tensor(1.), no_cuda).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return loss, torch.sum(grad**2)

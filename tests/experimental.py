import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f

import torch
import torch.nn as torchnn
import torch.nn.functional as torchf

np._set_promotion_state("weak_and_warn")


def main():

    # x = torch.arange(4)
    # x = x.float().requires_grad_()
    # z = torchf.sigmoid(x)
    #
    # def softmax(x, dim=-1):
    #     exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    #     return exp / torch.sum(exp, dim=dim, keepdim=True)
    #
    # def softmax_backward(p, grad):
    #     p = p.clone().detach()
    #     p = torch.flatten(p)
    #     diagonal = torch.diagflat(p)
    #     off_diagonal = torch.outer(p, p)
    #     return (diagonal - off_diagonal) * grad
    #
    # p = torchf.softmax(z, dim=-1)
    # p.backward(torch.ones_like(p))
    # print(f"AD Gradient: {x.grad}\n")
    #
    # grad = softmax_backward(p, 1)

    x = nura.randn(3, 3)
    y = x.clone()
    print(x)
    x @= y
    print(x)


if __name__ == "__main__":
    main()

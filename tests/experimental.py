import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np
import torch


def main():

    x = torch.arange(30).reshape(3, 10).float().requires_grad_()
    w = torch.nn.Parameter(torch.rand(10))
    optim = torch.optim.SGD(iter([w]), lr=1.0)

    w_ = w.clone().detach()
    y = x * w
    loss = y.sum()
    loss.backward()
    grad = w.grad
    optim.step()

    print(grad / 3)
    print(w_ - w)


if __name__ == "__main__":
    main()

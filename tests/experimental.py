import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch


def main():

    a = torch.rand(5, 4, 3, 2, requires_grad=True)
    b = torch.rand(2, 5, requires_grad=True)
    c = torch.matmul(a, b)
    c.backward(torch.ones_like(c))
    print(c.size())
    print(b.grad)

    a = deepnet.tensor(a.detach().numpy(), usegrad=True)
    b = deepnet.tensor(b.detach().numpy(), usegrad=True)
    c = deepnet.matmul(a, b)
    c.backward(deepnet.oneslike(c))
    print(c.dim)
    print(b.grad)


if __name__ == "__main__":
    main()

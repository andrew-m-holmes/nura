import numpy as np
import deepnet
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd, getperts
import torch


def main():
    
    a = torch.rand((4, 3), requires_grad=True)
    b = torch.rand((3, 2), requires_grad=False)
    c = torch.max(torch.matmul(a, b))
    print(c)
    c.backward()
    print(a.grad)

    a.grad.zero_()
    d = torch.min(torch.matmul(a, b))
    d.backward()
    print(a.grad)

    a = deepnet.tensor(a.detach().numpy(), usegrad=True)
    b = deepnet.tensor(b.numpy())
    d = deepnet.min(deepnet.matmul(a, b))
    d.backward()
    print(a.grad)

    a = deepnet.rand((4, 3), usegrad=True).float()
    b = deepnet.rand((3, 2), usegrad=False).float()
    c = deepnet.max(deepnet.matmul(a, b))
    print(c)
    c.backward()
    print(a.grad)

if __name__ == "__main__":
    main()

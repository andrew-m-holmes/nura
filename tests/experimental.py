import torch
import torch.autograd.functional as taf
import deepnet.autograd.functional as daf
import numpy as np
import deepnet


def main():

    a = deepnet.rand((2, 3, 4), use_grad=True)
    print(deepnet.is_contiguous(a))
    b = a[:, 1, 2:3]
    print(deepnet.is_contiguous(b))
    c = b.contiguous()
    print(deepnet.is_contiguous(c))
    print(c)
    c.backward(deepnet.ones_like(c))
    print(b.grad)

if __name__ == "__main__":
    main()

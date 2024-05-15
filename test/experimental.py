import nura
import nura.nn as nn
import nura.autograd.functional as f
import numpy as np
import torch


def main():

    a = nura.randn(3, usegrad=True)
    b = nura.randn(2, 3, 4, usegrad=True)
    c = nura.matmul(a, b)
    d = c.sum()
    d.backward()

    print(a.grad)
    print(b.grad)

    a = torch.from_numpy(a.data).requires_grad_()
    b = torch.from_numpy(b.data).requires_grad_()
    c = torch.matmul(a, b)
    d = c.sum()
    d.backward()

    print(a.grad)
    print(b.grad)


if __name__ == "__main__":
    main()

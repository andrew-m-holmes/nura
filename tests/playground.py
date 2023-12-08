import torch
import numpy as np
import deepnet
import deepnet.nn.functional as f


def main():
    a = deepnet.rand((1, 1, 1, 1, 1, 1, 1, 1, 1), use_grad=True)
    b = a.squeeze((0, 1, 2, 3, 5))
    print(b.dim())
    b.backward()
    print(a.grad)


if __name__ == "__main__":
    main()

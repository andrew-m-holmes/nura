import torch
import numpy as np
import deepnet
import deepnet.nn.functional as f


def main():
    a = deepnet.rand((4, 1), use_grad=True)
    b = deepnet.rand((3,), use_grad=True)
    c = a + b
    print(c.dim())
    c.backward(deepnet.zeros_like(c))


if __name__ == "__main__":
    main()

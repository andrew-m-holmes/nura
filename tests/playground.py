import torch
import numpy as np
import deepnet
import deepnet.nn.functional as f


def main():
    b = torch.tensor(4., requires_grad=True)
    print(b)
    print(b + 3)
    a = deepnet.tensor(3.)
    print(a)
    print(deepnet.typename(a))


if __name__ == "__main__":
    main()

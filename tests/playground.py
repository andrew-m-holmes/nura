import torch
import numpy as np
import deepnet
import deepnet.functional as f


def main():

    a = deepnet.tensor(5., use_grad=True)
    b = deepnet.tensor(7., use_grad=True)
    c = f.mul(a, 15)
    print(c)
    c.backward()
    print(a.grad)


if __name__ == "__main__":
    main()

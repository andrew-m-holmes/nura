import deepnet
import deepnet.functional as f
import numpy as np
import deepnet.autograd.functional as af
import torch
import torch.autograd.functional as taf

def main():

    def func(x, w, b):
        return f.matmul(x, w.t) + b

    x = deepnet.rand((5, 4), use_grad=True).float()
    w = deepnet.rand((3, 4), use_grad=True).float()
    b = deepnet.tensor(3, use_grad=True).float()
    v = deepnet.ones((5, 3)).float()

    output, vjp = af.vjp((x, w, b), v, func, use_graph=True)
    print(vjp)

if __name__ == "__main__":
    main()

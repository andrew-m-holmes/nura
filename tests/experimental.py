import jax
import numpy as np
import deepnet as dn
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd

def main():

    def linear(x, w, b):  
        return dn.matmul(x, w) + b

    inpt = dn.rand((64, 10), usegrad=False).float()
    w = dn.rand((10, 64), usegrad=True).float()
    b = dn.rand((64,), usegrad=True).float()

    pred = linear(inpt, w ,b)
    error = pred.sum()
    print(error)
    error.backward()
    print(w.grad)
    print(b.grad)

if __name__ == "__main__":
    main()

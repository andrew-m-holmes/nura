import numpy as np
import deepnet as dn
from deepnet.autograd.functional import grad, vjp


def main():

    def f(a, b, c):
        return a * b / c
    
    a = dn.rand(4).float()
    b = dn.rand(4).float()
    c = dn.rand(4).float()
    cotan = dn.rand(4).float()
    grads = vjp((a, b, c), cotan, f)
    print(grads)

if __name__ == "__main__":
    main()

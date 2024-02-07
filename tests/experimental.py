import numpy as np
import deepnet as dn
from deepnet.autograd.functional import grad, vjp


def main():
    
    a = dn.rand(4, usegrad=True, dtype=dn.float)
    b = dn.rand(4, usegrad=True, dtype=dn.float)
    c = dn.rand(1, usegrad=True, dtype=dn.float)

    def f(a, b, c):
        return a * b + c
    
    cot = dn.ones(4, dtype=dn.float)
    cots = vjp((a, b, c), cot, f)
    print(cots)

if __name__ == "__main__":
    main()

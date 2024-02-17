import numpy as np
import deepnet as dn
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd


def main():


    def f(a, b, c):
        return a * b + c

    a = dn.full(2, 2.0)
    b = dn.full(2, 1.)
    c = dn.tensor(8.)

    out, jac = jacrev((a, b, c), f)
    print(jac)

    out, jac = jacfwd((a, b, c), f)
    print(jac)

if __name__ == "__main__":
    main()

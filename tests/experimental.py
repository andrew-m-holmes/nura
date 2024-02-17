import numpy as np
import deepnet as dn
import jax
from deepnet.autograd.functional import vjp, jvp, grad, jacrev, jacfwd


def main():


    def f(a, b, c):
        return a * b + c

    a = dn.rand(2)
    b = dn.rand(2)
    c = dn.rand(1)

    out, jac = jacrev((a, b, c), f)
    print(jac)

    jacf = jax.jacrev(f)
    jac = jacf(a.data, b.data, c.data)
    print(jac)

    out, jac = jacfwd((a, b, c), f)
    print(jac)

if __name__ == "__main__":
    main()

import numpy as np
import deepnet as dn
from deepnet.autograd.functional import grad, vjp


def main():

    def f(a, b, c):
        return a * b + c

    a = dn.tensor([1.0, 1.0, 1.0], usegrad=True).float()
    b = dn.tensor([2.0, 2.0, 2.0], usegrad=True).float()
    c = dn.tensor([5.0], usegrad=True).float()
    v = dn.ones(3).float()
    grads = vjp((a, b, c), v, f)
    print(grads)


if __name__ == "__main__":
    main()

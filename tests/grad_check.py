import deepnet
import deepnet.nn.functional as f
import numpy as np
from deepnet.autograd.functional import vjp


def main():

    def func(a, b, c):
        d = a + b * c / 1000
        return d

    a = deepnet.tensor(1.)
    b = deepnet.tensor(2.)
    c = deepnet.tensor(3.)

    v = deepnet.tensor(2.)
    output, cotangents = vjp((a, b, c), v, func, use_graph=True)
    print(output, cotangents)


if __name__ == "__main__":
    main()

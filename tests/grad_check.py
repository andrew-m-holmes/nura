import deepnet
import deepnet.nn.functional as f
import numpy as np
from deepnet.autograd.functional import vjp


def main():

    def func(a, b, c):
        return a * b * c * 2

    a = deepnet.tensor(3)
    b = deepnet.tensor(4)
    c = deepnet.tensor(2)
    vector = deepnet.tensor(1)
    contangent = vjp((a, b, c), vector, func, keep_graph=True)
    print(contangent)
    contangent[0].backward()
    print(a.grad)


if __name__ == "__main__":
    main()

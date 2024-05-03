import numpy as np
import nura
import nura.nn as nn
from nura.autograd.function import Function
from nura.autograd.graph import constructgraph, topological


class Unbind(Function):

    @staticmethod
    def forward(context, a):
        context.save(a)
        return tuple(np.split(a.data, len(a.data), 0))

    @staticmethod
    def backward(context, *grad):
        grad = tuple(np.reshape(g.data, 1) for g in grad)
        return np.concatenate(grad, axis=0)


unbind = Unbind.apply


def main():

    a = nura.tensor(3.0, usegrad=True)
    b = nura.tensor(2.0, usegrad=True)
    c = a * b
    c.retaingrad()
    d = a * c
    d.retaingrad()
    d.backward()
    print(c.grad)
    print(d.grad)


if __name__ == "__main__":
    main()

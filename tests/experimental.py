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
        grad = tuple(g.data for g in grad)
        return np.concatenate(grad, axis=0)


unbind = Unbind.apply


def main():

    a = nura.tensor(3.0, usegrad=True)
    b = nura.tensor(2.0)
    c = a * b
    d = a * c
    d.backward()
    assert d.gradfn is not None


if __name__ == "__main__":
    main()

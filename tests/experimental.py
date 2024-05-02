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
        return np.concatenate(grad, axis=0)


unbind = Unbind.apply


def main():

    a = nura.tensor([1.0, 2.0, 3.0], usegrad=True)
    b, c, d = unbind(a)
    e = b * c
    f = e * d

    graph = constructgraph((f.gradfn,))
    sorted = topological(graph)
    print(sorted)


if __name__ == "__main__":
    main()

import numpy as np
import nura
import nura.nn as nn
import torch
from nura.autograd.function import Function
from nura.autograd.graph import construct_graph, topological


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

    a = torch.tensor([1.0, 2.0, 3], requires_grad=True)
    b, c, d = a.unbind()
    e = b * c
    f = e * d
    torch.autograd.backward(f)


if __name__ == "__main__":
    main()

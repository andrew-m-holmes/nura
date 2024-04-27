import numpy as np
import nura
import nura.nn as nn
import nura.nn.functional as f
from nura.autograd.function import Function


class MultiOutput(Function):

    @staticmethod
    def forward(context, a):
        context.save(a)
        return a.data * a.data, a.data + a.data, a.data - 1

    @staticmethod
    def backward(context, grad0, grad1, grad2):
        a = context.tensors()[0]
        return 2 * a.data * grad0.data + 2 * grad1.data + 1 * grad2.data


def multiout(a):
    return MultiOutput.apply(a)


def main():

    a = nura.tensor(3.0, usegrad=True)
    b, c, d = multiout(a)
    print(b, c, d)
    e = b + c + d
    print(e)
    e.backward()


if __name__ == "__main__":
    main()

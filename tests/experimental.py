import numpy as np
import nura
import nura.nn as nn
from nura.autograd.function import Function


class MulFunc(Function):

    @staticmethod
    def forward(context, a):
        context.save(a)
        return a.data * 2, a.data * 3

    @staticmethod
    def backward(context, grad0, grad1):
        return 2 * grad0.data + 3 * grad1.data


mulfunc = MulFunc.apply


def main():

    a = nura.rand().attached()
    b, c = mulfunc(a)
    d = b * c
    print(d.gradfn.nextfunctions())
    print(d.gradfn.nextfunctions()[0][0].nextfunctions())


if __name__ == "__main__":
    main()

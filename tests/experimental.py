import numpy as np
import nura
import nura.nn as nn
import torch
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

    a = torch.tensor([1.0, 2.0], requires_grad=True)
    b, c = a.unbind()
    print(b.grad_fn.next_functions[0][0])
    d = a + b
    print(d.grad_fn.next_functions[0][0] is (b.grad_fn.next_functions[0][0]))
    print(d.grad_fn.next_functions[1][0])


if __name__ == "__main__":
    main()

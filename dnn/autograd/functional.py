import numpy as np
from function import Function


class MyFunc(Function):

    @staticmethod
    def forward(ctx, *tensors):
        pass

    def backward(ctx, grad):
        pass

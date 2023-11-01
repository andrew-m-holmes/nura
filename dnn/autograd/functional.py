import numpy as np
from function import Function


class MyFunc(Function):

    @staticmethod
    def forward(ctx, *tensors):
        pass

    @staticmethod
    def backward(ctx, grad):
        pass

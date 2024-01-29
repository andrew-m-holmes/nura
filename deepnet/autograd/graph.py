import deepnet
import numpy as np
from . import mode

class Node:

    def __init__(self, fn, nxtfn):
        self.fn = fn
        self.nxtfn = nxtfn

    def apply(self, grad):
        return self.fn(grad)

    def __repr__(self):
        return self.fn.__class__.__name__

def accumgrad(tensor, grad):
    pass

def sumgrad(tensor, grad):
    sumdims = _get_sum_dims(tensor.dim, grad.dim, tensor.ndim, grad.ndim)
    keepdims = tensor.ndim == grad.ndim
    data = np.sum(grad.data, axis=sumdims, keepdims=keepdims)
    return grad.withstate(data=data)


def sumdims(tdim, gdim, tndim, gndim):
    paddim = np.pad(tdim, (gndim - tndim, 0), constant_values=0)
    mask = paddim != np.array(gdim)
    return np.where(mask)




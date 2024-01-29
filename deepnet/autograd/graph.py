import deepnet
import numpy as np
from . import mode

class Node:

    def __init__(self, ctx, nodefns=None):
        self.ctx = ctx
        self.nodefns = nodefns
        self.tensor = None

    def apply(self, grad):
        graddata = self.ctx.apply(grad)
        gradout = self._tograds(graddata)
        return gradout
    
    def link(self, tensor):
        self.tensor = tensor

    def _tograds(self, gradata): 
        if isinstance(gradata, np.ndarray):
            gradata = (gradata,)
        return tuple(deepnet.tensor(data) for data in gradata)
    
    def __repr__(self) -> str:
        return str(self.ctx.__class__.__name__)


class AccumGrad:

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def apply(self, grad):
        data = grad.data
        if self.tensor.dim != grad.dim and self.tensor.ndim <= grad.ndim:
            data = _reduce_graddata(self.tensor, grad)
        return data

    @staticmethod
    def accum(tensor, grad):
        if tensor.grad is None:
            tensor._grad = deepnet.ones_like(tensor)
        tensor._grad._data += grad._data

def _reduce_graddata(tensor, grad):
    sumdims = _sumdims(tensor.dim, grad.dim, tensor.ndim, grad.ndim)
    keepdims = tensor.ndim == grad.ndim
    return np.sum(grad.data, axis=sumdims, keepdims=keepdims)

def _sumdims(tdim, gdim, tndim, gndim):
    paddim = np.pad(tdim, (gndim - tndim, 0), constant_values=0)
    mask = paddim != np.array(gdim)
    return np.where(mask)


def _graphout(ctx, rawout):
    out = deepnet.tensor(rawout)
    if mode.gradon() and _candiff(ctx):
        out = _mutateout(ctx, out)
        out.backfn.link(out)
    return out

def _mutateout(ctx, out):
    nodefns = _get_nodefns(ctx.tensors())
    node = Node(ctx, nodefns)
    return out.withstate(backfn=node, diff=True, leaf=False)


def _candiff(ctx):
    return ctx.tensors() and any(tensor.diff for tensor in ctx.tensors())


def _get_nodefns(tensors):
    nodefns = []
    for tensor in tensors:
        nodefn = _get_nodefns_helper(tensor)
        nodefns.append(nodefn)
    return nodefns


def _get_nodefns_helper(tensor):
    if tensor.leaf:
        ctx = AccumGrad(tensor)
        node = Node(ctx, None)
        node.link(tensor)
        return node
    return tensor.backfn

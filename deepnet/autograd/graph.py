import deepnet
import numpy as np
from . import mode

class Node:

    def __init__(self, ctx, nodefns=None):
        self.ctx = ctx
        self.nodefns = nodefns

    def apply(self, grad):
        if self.nodefns:
            graddata = self.ctx.apply(grad)
            grads = self._postpro_apply(graddata)
            self._apply_nodefns(self.nodefns, grads)
        else:
            self.ctx.apply(grad)

    def _apply_nodefns(self, nodefns , grads):
        for nodefn, grad in zip(nodefns, grads):
            if nodefn is not None:
                nodefn.apply(grad)

    def _postpro_apply(self, gradata): 
        if isinstance(gradata, np.ndarray):
            gradata = (gradata,)
        return tuple(deepnet.tensor(data) for data in gradata)
    
    def __repr__(self) -> str:
        return str(self.ctx.__class__.__name__)


class AccumulateGrad:

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def apply(self, grad):
        if self.tensor.grad is None:
            self.tensor._grad = deepnet.zeros_like(self.tensor)
        data = grad.data
        if self.tensor.dim != grad.dim and self.tensor.ndim <= grad.ndim:
            data = _reduce_graddata(self.tensor, grad)
        self.tensor._grad._data += data


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
        return _mutateout(ctx, out)
    return out

def _mutateout(ctx, out):
    nodefns = _get_nodefns(ctx.tensors())
    node = Node(ctx, nodefns)
    return out.withstate(backfn=node, leaf=False)


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
        ctx = AccumulateGrad(tensor)
        return Node(ctx, None)
    return tensor.backfn

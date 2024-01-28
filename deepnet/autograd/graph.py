import numpy as np

import deepnet
from . import mode


class Graph:

    _graph = {}

    @classmethod
    def add(cls, tensor, node):
        cls._graph[tensor] = node

    @classmethod
    def delete(cls, tensor):
        cls._graph.pop(tensor)

    @classmethod
    def clear(cls):
        cls._graph.clear()


class Node:

    def __init__(self, ctx, nodefns=None):
        self.ctx = ctx
        self.nodefns = nodefns

    def apply(self, grad):
        grads = self.ctx.apply(grad)
        if self.nodefns:
            self._apply_nodefns(self.nodefns, grads)

    def _apply_nodefns(self, nodefns , grads):
        for nodefn, grad in zip(nodefns, grads):
            if nodefn is not None:
                nodefn.apply(grad)

    def __repr__(self) -> str:
        return str(self.ctx.__class__.__name__)


class AccumulateGrad:

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def apply(self, grad):
        if self.tensor.grad is None:
            self.tensor.grad = deepnet.zeros_like(self.tensor)
        grad_data = _process_grad_for_accumulate(self.tensor, grad)
        self.tensor.grad.data += grad_data.data


def _process_grad_for_accumulate(tensor, grad):
    if tensor.dim() != grad.dim() and tensor.ndim() <= grad.ndim():
        dims = _get_dims_to_sum(tensor.dim(), grad.dim())
        keepdims = tensor.ndim() == grad.ndim()
        return np.sum(grad.data, axis=dims, keepdims=keepdims)
    return grad.data


def _get_dims_to_sum(dim_0, dim_1):
    padded_dim_0 = np.pad(dim_0, (len(dim_1) - len(dim_0), 0), constant_values=0)
    mask = padded_dim_0 != np.array(dim_1)
    dims = tuple(i for i, bool_ in enumerate(mask) if bool_)
    return dims


def _graphout(ctx, rawout):
    out = deepnet.tensor(rawout)
    if mode.gradon():
        return _augmentout(ctx, out)
    return out

def _augmentout(ctx, out):
    if _candiff(ctx):
        nodefns = _get_nodefns(ctx.tensors())
        node = Node(ctx, nodefns)
        return out.withstate(backfn=node, leaf=False)
    return out


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
        return Node(ctx, nodefns=None)
    return tensor.nodefn

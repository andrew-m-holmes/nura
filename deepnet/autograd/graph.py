import numpy as np

import deepnet


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

    def __init__(self, ctx, backfns=None):
        self.ctx = ctx
        self.backfns = backfns

    def apply(self, grad):
        next_grads = self.ctx.apply(grad)
        if self.backfns:
            self._apply_backfns(self.backfns, next_grads)

    def _apply_backfns(self, backfns, next_grads):
        for next_function, grad in zip(backfns, next_grads):
            if next_function is not None:
                next_function.apply(grad)

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


def _pass_to_graph(ctx, output):
    output = _pass_for_reverse_ad(ctx, output)
    return output


def _fwdout(ctx, output):
    tan = ctx.apply_jvp()
    return None
    output._set_dual_state(tangent_out, True)
    return output


def _revout(ctx, output):
    if _diff_ctx(ctx):
        backfns = _get_backfns(ctx.tensors())
        node = Node(ctx, backfns)
        output = output.withattrs(backfn=node, leaf=False)
    return output

def _diff_ctx(ctx):
    if ctx.tensors():
        return any(tensor.diff for tensor in ctx.tensors())
    return False


def _get_backfns(tensors):
    backfns = []
    for tensor in tensors:
        next_function = _get_backfns_helper(tensor)
        backfns.append(next_function)
    return backfns

def _get_backfns_helper(tensor):
    if tensor.leaf:
        ctx = AccumulateGrad(tensor)
        return Node(ctx, backfns=None)
    return tensor.backfn

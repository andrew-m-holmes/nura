import numpy as np
import deepnet
from deepnet import Tensor, DualTensor
from .mode import Autograd


class Node:

    def __init__(self, context, next_functions=None):
        self.context = context
        self.next_functions = next_functions

    def apply(self, grad):
        next_grads = self.context.apply(grad)
        if self.next_functions:
            self._apply_next_functions(
                self.next_functions, next_grads)

    def _apply_next_functions(self, next_functions, next_grads):
        next_grads = _preprocess_grad_output(next_grads)
        for next_function, grad in zip(next_functions, next_grads):
            if next_function is not None:
                next_function.apply(grad)

    @classmethod
    def with_context(cls, context, next_functions):
        return cls(context, next_functions)

    def __repr__(self) -> str:
        return str(self.context.__class__.__name__)


class AccumulateGrad:

    def __init__(self, tensor) -> None:
        self._tensor = tensor

    def apply(self, grad):
        if self._tensor.grad is None:
            self._tensor.grad = deepnet.zeros_like(self._tensor)
        self._tensor.grad.data += grad.data

    def tensor(self):
        return self._tensor

    @classmethod
    def with_tensor(cls, tensor):
        return cls(tensor)


def _pass_to_graph(context, output):
    if Autograd.grad_enabled():
        output = _pass_for_reverse_ad(context, output)
    if Autograd.forward_ad_enabled():
        output = _pass_for_forward_ad(context, output)
    return output


def _pass_for_reverse_ad(context, output):
    if _context_has_grad_tensors(context):
        saved_tensors = _preprocess_for_reverse_ad(context.saved_tensors())
        next_functions = _get_next_functions(saved_tensors)
        node = Node.with_context(context, next_functions)
        output._set_grad_state(use_grad=True, grad_fn=node, is_leaf=False)
    return output


def _pass_for_forward_ad(context, output):
    tangents = [dual_tensor.tangent for dual_tensor in context.saved_tensors()]
    tangent_out = context.apply_jvp(*tangents)
    output = deepnet.dual_tensor(output, tangent_out)
    return output


def _context_has_grad_tensors(context):
    if context.saved_tensors():
        return any(tensor.use_grad for tensor in context.saved_tensors())
    return False


def _get_next_functions(saved_tensors):
    next_functions = []
    for tensor in saved_tensors:
        next_function = _get_next_functions_helper(
            tensor)
        next_functions.append(next_function)
    return tuple(next_functions)


def _get_next_functions_helper(tensor):
    if tensor.is_leaf and tensor.use_grad:
        context = AccumulateGrad.with_tensor(tensor)
        return Node.with_context(context, next_functions=None)
    return tensor.grad_fn


def _preprocess_for_reverse_ad(saved_tensors):
    processed_tensors = []
    for tensor in saved_tensors:
        if isinstance(tensor, DualTensor):
            processed_tensors.append(tensor.primal)
        else:
            processed_tensors.append(tensor)
    return processed_tensors


def _preprocess_grad_output(grad):
    if isinstance(grad, Tensor):
        grad = (grad,)
    return grad

import numpy as np
from deepnet.tensor import Tensor


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
            self._tensor.grad = Tensor(
                np.zeros_like(self._tensor.data))
        self._tensor.grad.data += grad.data

    def tensor(self):
        return self._tensor

    @classmethod
    def with_tensor(cls, tensor):
        return cls(tensor)


def pass_to_graph(context, output):
    if any(tensor.use_grad for tensor in context.saved_tensors()):
        next_functions = _get_next_functions(context)
        output.grad_fn = Node(context, next_functions)
        output.use_grad = True
        output.is_leaf = False


def _get_next_functions(context):
    if hasattr(context, "saved_tensors"):
        next_functions = []
        for tensor in context.saved_tensors():
            next_function = _get_next_functions_helper(
                tensor)
            next_functions.append(next_function)
        return tuple(next_functions)


def _get_next_functions_helper(tensor):
    if tensor.use_grad and tensor.is_leaf:
        context = AccumulateGrad.with_tensor(tensor)
        return Node.with_context(context, next_functions=None)
    if tensor.use_grad:
        return tensor.grad_fn
    return None

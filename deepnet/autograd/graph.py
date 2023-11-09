import numpy as np
from deepnet.tensor import Tensor


class Node:

    def __init__(self, backward_fn, context=None):
        self.backward_fn = backward_fn
        self.context = context
        self.next_functions = self.get_next_functions(context) \
            if context is not None else None

    def apply(self, grad):
        grads = self.backward_fn.apply(
            self.context, grad)
        if self.next_functions:
            self._apply_next_functions(
                self.next_functions, grads)

    def _apply_next_functions(self, next_functions, grads):
        for next_function, grad in zip(next_functions, grads):
            if next_function is not None:
                next_function.apply(grad)

    def get_next_functions(self, context):
        if context is not None:
            next_functions = []
            for tensor in context.stored():
                next_function = self._get_next_functions_helper(
                    tensor)
                next_functions.append(next_function)
            return next_functions

    def _get_next_functions_helper(self, tensor):
        if tensor.use_grad and tensor.is_leaf:
            return Node(AccumulateGrad(tensor), None)
        if tensor.use_grad:
            return tensor.grad_fn
        return None

    def __repr__(self) -> str:
        return repr(self.backward_fn)


class AccumulateGrad:

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def apply(self, context, grad):
        if self.tensor.grad is None:
            self.tensor.grad = Tensor(
                np.zeros_like(self.tensor.data))
        self.tensor.grad.data += grad.data

    def __repr__(self) -> str:
        return self.__class__.__name__


def pass_to_graph(backward_fn, context, output):
    if any(tensor.use_grad for tensor in context.stored()):
        output.grad_fn = Node(backward_fn, context)
        output.use_grad = True
        output.is_leaf = False

import numpy as np
from deepnet.tensor import Tensor


class Node:

    def __init__(self, context):
        self.context = context
        self.next_functions = self.get_next_functions(context)

    def apply(self, grad):
        next_grad = self.context.apply(grad)
        if self.next_functions:
            self._apply_next_functions(
                self.next_functions, next_grad)

    def _apply_next_functions(self, next_functions, next_grads):
        for next_function, grad in zip(next_functions, next_grads):
            if next_function is not None:
                next_function.apply(grad)

    def get_next_functions(self, context):
        if hasattr(context, "saved_tensors"):
            next_functions = []
            for tensor in context.saved_tensors():
                next_function = self._get_next_functions_helper(
                    tensor)
                next_functions.append(next_function)
            return next_functions

    def _get_next_functions_helper(self, tensor):
        if tensor.use_grad and tensor.is_leaf:
            return Node(AccumulateGrad.with_tensor(tensor))
        if tensor.use_grad:
            return tensor.grad_fn
        return None

    def __repr__(self) -> str:
        return str(self.context.__class__.__name__)


class AccumulateGrad:

    def __init__(self, tensor) -> None:
        self.tensor = tensor

    def apply(self, grad):
        if self.tensor.grad is None:
            self.tensor.grad = Tensor(
                np.zeros_like(self.tensor.data))
        self.tensor.grad.data += grad.data

    @classmethod
    def with_tensor(cls, tensor):
        return cls(tensor)


def pass_to_graph(context, output):
    if any(tensor.use_grad for tensor in context.saved_tensors()):
        output.grad_fn = Node(context)
        output.use_grad = True
        output.is_leaf = False

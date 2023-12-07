import deepnet
import deepnet.nn.functional as f
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
        self.tensor = tensor

    def apply(self, grad):
        if self.tensor.grad is None:
            self.tensor.grad = deepnet.zeros_like(self.tensor)
        grad = self._preprocess_for_grad_accumulation(grad)
        self.tensor.grad.data += grad.data

    def _preprocess_for_grad_accumulation(self, grad):
        if grad.dim() != self.tensor.dim():
            if _is_scalar(self.tensor):
                grad = f.sum(grad)
            else:
                dims = _get_dims_to_sum(self.tensor.dim(), grad.dim())
                grad = f.sum(grad, dims, keepdims=True)
                print(grad.dim())
        return grad

    @classmethod
    def with_tensor(cls, tensor):
        return cls(tensor)


def _is_scalar(tensor):
    return tensor.ndim() == 0


def _get_dims_to_sum(dim_1, dim_2):
    dim_1 = _pad_dim(dim_1, dim_2)
    dims_to_sum = [i for i in range(
        len(dim_2)) if dim_2[i] > dim_1[i]]
    print(dim_1, dim_2, dims_to_sum)
    return tuple(dims_to_sum)


def _pad_dim(dim_1, dim_2):
    length = len(dim_2) - len(dim_1)
    return [1] * length + list(dim_1)


def _pass_to_graph(context, output):
    if Autograd.grad_enabled():
        output = _pass_for_reverse_ad(context, output)
    if Autograd.forward_ad_enabled():
        output = _pass_for_forward_ad(context, output)
    return output


def _pass_for_reverse_ad(context, output):
    if _context_has_grad_tensors(context):
        saved_tensors = _preprocess_for_reverse_ad(
            context.saved_tensors())
        next_functions = _get_next_functions(saved_tensors)
        node = Node.with_context(context, next_functions)
        output._set_grad_state(
            use_grad=True, grad_fn=node, is_leaf=False)
    return output


def _pass_for_forward_ad(context, output):
    tangents = [dual_tensor.tangent
                for dual_tensor in context.saved_tensors()]
    tangent_out = context.apply_jvp(*tangents)
    output = deepnet.dual_tensor(output, tangent_out)
    return output


def _context_has_grad_tensors(context):
    if context.saved_tensors():
        return any(tensor.use_grad
                   for tensor in context.saved_tensors())
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
        if deepnet.is_dual_tensor(tensor):
            processed_tensors.append(tensor.primal)
        else:
            processed_tensors.append(tensor)
    return processed_tensors


def _preprocess_grad_output(grad):
    if deepnet.is_tensor(grad):
        grad = (grad,)
    return grad

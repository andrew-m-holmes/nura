from dnn.tensor import Tensor
compute_grads = True


class Node:

    def __init__(self, ctx, output, backward_fn):
        self.ctx = ctx
        self.backward_fn = backward_fn
        self.next_functions = self._get_next_functions(ctx)
        self._setup_output(output, ctx)

    def backward(self, grad):
        tensor_grads = self.backward_fn(self.ctx, grad)
        for tensor_grad, next_function in zip(tensor_grads, self.next_functions):
            next_grad = Tensor(tensor_grad.data * grad.data)
            next_function.backward(next_grad)

    def _get_next_functions(self, ctx):
        next_functions = []
        for tensor in ctx.saved_tensors:
            next_function = self._get_next_function_helper(
                tensor)
            next_functions.append(next_function)
        return next_functions

    def _get_next_function_helper(self, tensor):
        if tensor.use_grad and tensor.is_leaf:
            return AccumulateGrad(tensor)
        elif tensor.use_grad:
            return tensor.grad_fn  # point to another Node
        return None

    def _setup_output(self, output, ctx):
        if any(tensor.use_grad for tensor in ctx.saved_tensors):
            output.grad_fn = self
            output.use_grad = True

    def __repr__(self) -> str:
        return f"{str(self.backward_fn).split()[1]}"


class AccumulateGrad:

    def __init__(self, tensor):
        self.tensor = tensor

    def backward(self, grad):
        accumulate_grad(self.tensor, grad)

    def __repr__(self) -> str:
        return self.__class__.__name__


def accumulate_grad(tensor, grad):
    if tensor.grad is None:
        # TODO make it an zeros tensor w/ same shape as tensor
        tensor.grad = Tensor([0.])
    tensor.grad.data += grad.data

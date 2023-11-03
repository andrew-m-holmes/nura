

class Node:

    def __init__(self, backward_fn, ctx, output_tensor) -> None:
        self._backward_fn = backward_fn
        self.ctx = ctx
        self.output = output_tensor

        if self._compute_grads(ctx):
            self.next_functions = self._generate_next_functions(
                ctx)

    def _compute_grads(self, ctx):
        return any(tensor.use_grad for tensor in ctx)

    def _generate_next_functions(self, ctx):
        next_functions = []
        for tensor in ctx:
            if tensor.use_grad and tensor.is_leaf:
                # TODO add accumulate grad
                next_functions.append("AccumulateGrad")
            elif tensor.is_leaf:
                next_functions.append(None)
            elif tensor.use_grad:
                next_functions.append(tensor.grad_fn)
        return next_functions


def generate_node(backward_fn, ctx):
    Node(backward_fn, ctx)

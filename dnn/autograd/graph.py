
graph_enabled = True


class Node:

    def __init__(self, ctx, backward_fn):
        self.ctx = ctx
        self.baackward_fn = backward_fn

    def apply(self, tensor, grad):
        grads = []
        for tensor in self.ctx.saved():
            pass


class Node:

    def __init__(self, ctx, backward_fn):
        self.ctx = ctx
        self.backward_fn = backward_fn
        self.next_functions = self.generate_next_functions(ctx)

    def generate_next_functions(self, ctx):
        next_functions = []
        for tensor in ctx.saved():
            next_function = self.tensor_next_function(tensor)
            next_functions.append(next_function)
        return next_functions

    def tensor_next_function(self, tensor):
        if tensor.is_leaf and tensor.use_grad:
            return AccumulateGrad(tensor)
        if tensor.use_grad:
            return tensor.grad_fn
        return None


class AccumulateGrad:

    def __init__(self, tensor):
        pass

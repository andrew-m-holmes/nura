from graph import Node


class Context:

    def save_for_forward(self, *tensors):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        assert (isinstance(tensors) for tensor in tensors)
        self.saved_tensors = tensors


class Function:

    @staticmethod
    def forward(ctx, *tensors):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError

    @classmethod
    def apply(cls, *tensors):
        ctx = Context()
        output = cls.forward(ctx, *tensors)
        output.grad_fn = Node(ctx, output, cls.backward)
        return output

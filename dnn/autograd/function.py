from graph import generate_node


class Context:

    def save_for_forward(self, *tensors):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        self.saved_backward_tensors = tensors


class Function:

    @staticmethod
    def forward(ctx, *tensors):
        pass

    @staticmethod
    def backward(ctx, grad):
        pass

    @classmethod
    def apply(cls, *tensors):
        ctx = Context()
        output = cls.forward(ctx, *tensors)
        generate_node(cls.backward, ctx)
        return output

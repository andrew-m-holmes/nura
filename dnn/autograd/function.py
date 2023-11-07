from graph import Node


class Context:

    def __init__(self) -> None:
        self.saved_tensors = None

    def save_for_forward(self, *tensors):
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        assert (isinstance(tensors) for tensor in tensors)
        self.saved_tensors = tensors

    def saved(self):
        return self.saved_tensors


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
        Node(ctx, output, cls.backward)
        return output

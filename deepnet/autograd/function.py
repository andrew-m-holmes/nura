from deepnet.autograd.graph import pass_to_graph
from deepnet.tensor import Tensor


class Context:

    def __init__(self) -> None:
        self.stored_tensors = None

    def store(self, *tensors, **kwargs):
        assert self.stored_tensors is None, \
            "Function Context is already storing Tensors"
        assert all(isinstance(tensor, Tensor)
                   for tensor in tensors), \
            "Function Context only accepts Tensors"
        self.stored_tensors = tensors
        self._set_attrs(**kwargs)

    def stored(self):
        return self.stored_tensors

    def _set_attrs(self, **kwargs):
        for key, value in kwargs.items():
            if key not in dir(self):
                setattr(self, key, value)


class BackwardFunction:

    def __init__(self, forward_cls):
        self._forward_cls = forward_cls
        self._backward_fn = forward_cls.backward

    def apply(self, *args):
        return self._backward_fn(*args)

    def __repr__(self) -> str:
        return repr(self._backward_fn).split()[1]


class Function:

    @staticmethod
    def forward(*tensors):
        raise NotImplementedError

    @staticmethod
    def backward(context, grad):
        raise NotImplementedError

    @staticmethod
    def create_context(context, *tensors):
        raise NotImplementedError

    @classmethod
    def apply(cls, *tensors):
        context = cls.create_context(Context(), *tensors)
        output = cls.forward(*tensors)
        pass_to_graph(BackwardFunction(cls), context, output)
        return output

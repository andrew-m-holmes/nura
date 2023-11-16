from deepnet import Tensor
from deepnet.autograd.graph import pass_to_graph


class Context:

    def __init__(self) -> None:
        self._saved_tensors = None

    def save_for_backward(self, *tensors):
        assert self._saved_tensors is None, \
            "Function Context is already storing Tensors"
        assert all(isinstance(tensor, Tensor)
                   for tensor in tensors), \
            "Function Context only accepts Tensors"
        self._saved_tensors = tensors

    def saved_tensors(self):
        return self._saved_tensors


class BackwardFunction(Context):

    def apply(self, *args):
        backward_fn = self._forward_cls.backward
        return backward_fn(self, *args)


class FunctionMeta(type):

    def __init__(cls, name, bases, attrs):
        backward_cls = type(name + "Backward",
                            (BackwardFunction, ), {"_forward_cls": cls})
        cls._backward_cls = backward_cls
        super().__init__(name, bases, attrs)


class Function(metaclass=FunctionMeta):

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
        context = cls.create_context(cls._backward_cls(), *tensors)
        output = cls.forward(*tensors)
        pass_to_graph(context, output)
        return output

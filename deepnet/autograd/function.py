from deepnet import Tensor, DualTensor
from .graph import _pass_to_graph
from typing import Tuple, Union


class Context:

    def __init__(self) -> None:
        self._saved_tensors = None

    def save_tensors(self, *tensors):
        self._saved_tensors = tensors

    def saved_tensors(self) -> Tuple[Union[Tensor, DualTensor], ...]:
        return self._saved_tensors


class BackwardFunction(Context):

    def apply(self, *args):
        backward_fn = self._forward_cls.backward
        return backward_fn(self, *args)

    def apply_jvp(self, *args):
        jvp_fn = self._forward_cls.jvp
        return jvp_fn(self, *args)


class FunctionMeta(type):

    def __init__(cls, name, bases, attrs):
        backward_cls = type(name + "Backward",
                            (BackwardFunction, ), {"_forward_cls": cls})
        cls._backward_cls = backward_cls
        super().__init__(name, bases, attrs)


class Function(metaclass=FunctionMeta):

    @staticmethod
    def forward(context, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(context, grad):
        raise NotImplementedError

    @staticmethod
    def jvp(context, *tangents):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        assert all(isinstance(arg, Tensor) or isinstance(arg, DualTensor)
                   for arg in args), \
            f"Invalid argument(s): {args}, Function.apply() only accepts Tensor and DualTensor objects"
        context = cls._backward_cls()
        output = cls.forward(context, *args, **kwargs)
        output = _pass_to_graph(context, output)
        return output

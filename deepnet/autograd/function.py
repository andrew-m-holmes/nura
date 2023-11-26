from deepnet import Tensor
from .mode import Grad
from .forward_autograd import DualTensor, _pass_for_forward_autograd
from .graph import _pass_to_graph
from typing import Tuple


class Context:

    def __init__(self) -> None:
        self._saved_tensors = None

    def save_tensors(self, *tensors):
        assert self._saved_tensors is None, \
            "Function Context is already storing Tensors"
        assert all(isinstance(tensor, Tensor) or isinstance(tensor, DualTensor)
                   for tensor in tensors), \
            "Function Context only accepts Tensors or DualTensors"
        self._saved_tensors = tensors

    def saved_tensors(self) -> Tuple[Tensor, ...]:
        return self._saved_tensors


class BackwardFunction(Context):

    def apply(self, *args):
        backward_fn = self._forward_cls.backward \
            if Grad.in_reverse_mode() else self._forward_cls.jvp
        return backward_fn(self, *args)


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
    def jvp(context, grad):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        args = _cast_args_for_forward(*args)
        context = cls._backward_cls()
        output = cls.forward(context, *args, **kwargs)
        if Grad.enabled():
            if Grad.in_reverse_mode():
                output = _pass_to_graph(context, output)
            else:
                output = _pass_for_forward_autograd(context, output, *args)
        return output


def _cast_args_for_forward(*args):
    t_cls = Tensor if Grad.in_reverse_mode() else DualTensor
    casted = []
    for arg in args:
        if not isinstance(arg, t_cls):
            casted.append(t_cls(arg))
        else:
            casted.append(arg)
    return tuple(casted)

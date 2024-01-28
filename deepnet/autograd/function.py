import deepnet
from . import graph as graph
from deepnet import Tensor
from typing import Tuple, Union


class Context:

    def __init__(self) -> None:
        self._tensors = None

    def save(self, *tensors: Tensor):
        self._tensors = tensors

    def tensors(self) -> Tuple[Tensor, ...]:
        return self._tensors

class BackwardFunction(Context):

    def apply(self, *args):
        backfn = self.fncls.backward
        return backfn(self, *args)

    def apply_jvp(self):
        jvp_fn = self.fncls.jvp
        return jvp_fn(self)


class FunctionMeta(type):

    def __init__(cls, name, bases, attrs):
        backcls = type(
            name + "Backward", (BackwardFunction,),
            {"fncls": cls})
        cls.backcls = backcls
        super().__init__(name, bases, attrs)


class Function(metaclass=FunctionMeta):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError

    @staticmethod
    def jvp(ctx):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        ctx = cls.backcls()
        output = cls.forward(ctx, *args, **kwargs)
        output = graph._pass_to_graph(ctx, output)
        return output

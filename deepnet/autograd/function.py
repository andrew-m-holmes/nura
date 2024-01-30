import numpy as np
from . import graph
import deepnet
from deepnet.tensors import Tensor
from typing import Tuple, Union


class Context:

    def __init__(self) -> None:
        self._tensors = None

    def save(self, *tensors: Tensor):
        self._tensors = tensors

    def tensors(self) -> Union[Tensor, Tuple[Tensor, ...], None]:
        if self._tensors is None:
            return None
        return self._tensors if len(self._tensors) > 1 else self._tensors[0]

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
            name.lower(), (BackwardFunction,),
            {"fncls": cls})
        cls.backcls = backcls
        super().__init__(name, bases, attrs)


class Function(metaclass=FunctionMeta):

    @staticmethod
    def forward(ctx, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError

    @staticmethod
    def jvp(ctx, *grads) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        ctx = cls.backcls()
        rawout = cls.forward(ctx, *args, **kwargs)
        irout = deepnet.tensor(rawout)
        out = graph.genout(irout, ctx)
        return out

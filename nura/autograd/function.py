import nura
from nura.tensors import Tensor
from nura.autograd.graph import genout
from typing import Tuple, Any, Optional, Dict, Union
from numpy import ndarray


class Context:

    def __init__(self) -> None:
        self._tensors: Optional[Tuple[Tensor, ...]] = None
        self._dict: Optional[Dict[Any, Any]] = None

    def save(self, *tensors: Tensor):
        for t in tensors:
            t.mutate(graph=t.graph + 1)
        self._tensors = tensors

    def tensors(self) -> Tuple[Tensor, ...]:
        return self._tensors if self._tensors else ()

    def usesgrad(self) -> bool:
        if self._tensors is None:
            return False
        return any(t.usegrad for t in self._tensors) and all(
            t.gradtensor for t in self._tensors
        )

    def __setitem__(self, key: Any, value: Any):
        if self._dict is None:
            self._dict = dict()
        self._dict[key] = value

    def __getitem__(self, key: Any) -> Any:
        assert self._dict is not None
        return self._dict[key]

    def __del__(self):
        for t in self.tensors():
            t.mutate(graph=t.graph - 1)

    def __repr__(self) -> str:
        return self.__class__.__name__


class Function:

    @staticmethod
    def forward(context: Context, *args: Any, **kwargs: Any) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(context: Context, grad: Tensor) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @staticmethod
    def tangent(context: Context, *grad: Tensor) -> ndarray:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any:
        context = Context()
        rawout = cls.forward(context, *args, **kwargs)
        irout = nura.tensor(rawout)
        out = genout(irout, cls, context)
        return out

    @classmethod
    def name(cls) -> str:
        return cls.__name__

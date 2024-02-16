import deepnet
from deepnet.tensors import Tensor
from deepnet.autograd.graph import genout
from typing import Tuple, Union, Any, Optional, Dict
from numpy import ndarray


class Context:

    def __init__(self) -> None:
        self._tensors: Optional[Tuple[Tensor, ...]] = None
        self._dict: Optional[Dict[Any, Any]] = None

    def save(self, *tensors: Tensor):
        self._tensors = tensors

    def tensors(self) -> Tuple[Tensor, ...]:
        return self._tensors if self._tensors else ()

    def __setitem__(self, key: Any, value: Any):
        if self._dict is None:
            self._dict = dict()
        self._dict[key] = value

    def __getitem__(self, key: Any) -> Any:
        assert self._dict is not None
        return self._dict[key]

    def __repr__(self) -> str:
        return self.__class__.__name__


class Function:

    @staticmethod
    def forward(context: Context, *args: Union[Tensor, Any], **kwargs: Any) -> ndarray:
        raise NotImplementedError

    @staticmethod
    def backward(context: Context, grad: Tensor) -> Union[Tuple[ndarray, ...], ndarray]:
        raise NotImplementedError

    @staticmethod
    def tangent(context: Context, *grad: Tensor) -> ndarray:
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: Union[Tensor, Any], **kwargs: Any) -> Tensor:
        context = Context()
        rawout = cls.forward(context, *args, **kwargs)
        irout = deepnet.tensor(rawout)
        out = genout(irout, cls, context)
        return out

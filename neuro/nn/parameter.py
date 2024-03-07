import neuro
from typing import Type, Optional, Any
from numpy import ndarray
from neuro.tensors import Tensor
from neuro.types import dtype


class Parameter(Tensor):

    def __init__(self, data: ndarray, dtype: Type[dtype]) -> None:
        super().__init__(
            data, usegrad=True, grad=None, backfn=None, leaf=True, _dtype=dtype
        )

    def to(self, dtype: Optional[Type[dtype]] = None) -> "Parameter":
        a = super().to(dtype)
        return parameter(a)

    def zeroedgrad(self):
        return parameter(super().zeroedgrad())

    def mutated(self, **attrs: Any) -> "Parameter":
        return parameter(super().mutated(**attrs))


def parameter(a: Tensor, dtype: Optional[Type[dtype]] = None) -> Parameter:
    validtypes = (neuro.half, neuro.float, neuro.double)
    if dtype is None:
        dtype = a.dtype
    assert dtype in validtypes
    return Parameter(dtype.numpy(a.data), dtype)

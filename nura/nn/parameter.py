import nura
from nura.types import dtype
from nura.tensors import Tensor
from nura.autograd.graph import Node
from typing import Optional, Type
from numpy import ndarray


class Parameter(Tensor):

    _gradtensor = True

    def __init__(
        self,
        data: ndarray,
        usegrad=True,
        grad: Optional["Tensor"] = None,
        backfn: Optional[Node] = None,
        leaf=True,
    ) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf)

    def to(self, dtype: Type[dtype]):
        return param(super().to(dtype), self.usegrad, dtype)

    def __repr__(self) -> str:
        return super().__repr__().replace("tensor", "param")


def param(a: Tensor, usegrad=True, dtype: Optional[Type[dtype]] = None):
    validtypes = (nura.half, nura.float, nura.double)
    if dtype is None:
        assert a.dtype is not None
        dtype = a.dtype
    assert dtype in validtypes, f"Parameter cannot type {dtype.name()}"
    data = dtype.numpy(a.data)
    p = Parameter(data, usegrad, None, None, True)
    p._dtype = dtype
    return p
import nura.types as types
from nura.types import dtype
from nura.tensors import Tensor
from nura.autograd.graph import Node
from typing import Optional, Type
from numpy import ndarray


class Parameter(Tensor):

    def __init__(
        self,
        data: ndarray,
        usegrad=True,
        grad: Optional["Tensor"] = None,
        gradfn: Optional[Node] = None,
        leaf=True,
    ) -> None:
        super().__init__(data, usegrad, grad, gradfn, leaf)

    def to(self, dtype: Type[dtype]):
        return parameter(super().to(dtype), self.usegrad, dtype)

    def __repr__(self) -> str:
        return super().__repr__().replace("tensor", "param")


def parameter(a: Tensor, usegrad=True, dtype: Optional[Type[dtype]] = None):
    validtypes = (types.half, types.float, types.double)
    if dtype is None:
        dtype = a.dtype
    if dtype not in validtypes:
        raise ValueError(
            f"Parameters can only be of floating-point types, received {dtype.name()}"
        )
    data = dtype.numpy(a.data)
    p = Parameter(data, usegrad, None, None, True)
    return p

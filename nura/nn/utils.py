import nura
from nura.tensors import Tensor
from nura.types import dtype
from typing import Optional, Type


def computedecay(tensor: Tensor, grad: Tensor, decay: float) -> Tensor:
    return grad + tensor * decay


def xavier(
    inputdim: int,
    outputdim: int,
    usegrad: bool = False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if dtype is None:
        dtype = nura.float

    var = pow(6 / (inputdim + outputdim), 0.5)
    return nura.uniform(-var, var, (outputdim, inputdim), usegrad, dtype)


def he(
    inputdim: int,
    outputdim: int,
    usegrad: bool = False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if dtype is None:
        dtype = nura.float

    std = pow(2 / inputdim, 0.5)
    dist = nura.randn(outputdim, inputdim, dtype=dtype)
    tensor = dist * std
    tensor.usegrad = usegrad
    return tensor

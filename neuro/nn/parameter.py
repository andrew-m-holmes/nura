from typing import Type, Union
from numpy import ndarray
from neuro.tensors import Tensor
from neuro.types import dtype


class Parameter(Tensor):

    def __init__(
        self,
        data: Union[Tensor, ndarray],
        _dtype: Type[dtype],
    ) -> None:
        data = paramdata(data, _dtype)
        super().__init__(
            data, usegrad=True, grad=None, backfn=None, leaf=True, _dtype=_dtype
        )

    def to(self, dtype: Type[dtype]) -> "Parameter":
        return Parameter(self.data, dtype)


def paramdata(data: Union[ndarray, Tensor], _dtype: Type[dtype]) -> ndarray:
    if isinstance(data, Tensor):
        data = data.data
    return _dtype.numpy(data)

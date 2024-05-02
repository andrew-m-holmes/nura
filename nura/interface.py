import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from typing import Type, Any, Tuple, Union, Iterable, List, Optional, Self

pyint = int
pyfloat = float
pybool = bool
inf = np.inf
dim = Tuple[int, ...]
dimlike = Union[Tuple[int, ...], int]
Scalar = Union[float, int, bool]
Tensorlike = Union[Iterable[Any], Scalar]


class dtype(ABC):
    _wrapping = None

    @classmethod
    @abstractmethod
    def numpy(cls, data) -> ndarray: ...

    @classmethod
    @abstractmethod
    def name(cls) -> str: ...


class Node(ABC):

    @abstractmethod
    def __init__(self) -> None: ...


Dtype = dtype
Dim = dim
Dimlike = dimlike


class Tensor(ABC):
    def __init__(
        self,
        data: ndarray,
        usegrad: bool,
        grad: Optional["Tensor"],
        gradfn: Optional[Node],
        leaf: bool,
    ) -> None: ...

    @property
    @abstractmethod
    def data(self) -> ndarray: ...

    @property
    @abstractmethod
    def dim(self) -> dim: ...

    @property
    @abstractmethod
    def ndim(self) -> int: ...

    @property
    @abstractmethod
    def nelem(self) -> int: ...

    @property
    @abstractmethod
    def usegrad(self) -> bool: ...

    @property
    @abstractmethod
    def grad(self) -> Optional["Tensor"]: ...

    @property
    @abstractmethod
    def gradfn(self) -> Optional[Node]: ...

    @property
    @abstractmethod
    def leaf(self) -> bool: ...

    @property
    @abstractmethod
    def index(self) -> int: ...

    @property
    @abstractmethod
    def version(self) -> int: ...

    @property
    @abstractmethod
    def dtype(self) -> Type[dtype]: ...

    @property
    @abstractmethod
    def gradtensor(self) -> bool: ...

    @property
    @abstractmethod
    def T(self) -> "Tensor": ...

    @abstractmethod
    def item(self) -> Scalar: ...

    @abstractmethod
    def list(self) -> List[Any]: ...

    @abstractmethod
    def backward(self, grad: Optional["Tensor"] = None) -> None: ...

    @abstractmethod
    def cleargrad(self) -> None: ...

    @abstractmethod
    def clearedgrad(self) -> "Tensor": ...

    @abstractmethod
    def zerograd(self) -> None: ...

    @abstractmethod
    def zeroedgrad(self) -> "Tensor": ...

    @abstractmethod
    def attach(self) -> None: ...

    @abstractmethod
    def attached(self) -> "Tensor": ...

    @abstractmethod
    def detach(self) -> None: ...

    @abstractmethod
    def detached(self) -> "Tensor": ...

    @abstractmethod
    def mutate(self, **attrs: Any) -> None: ...

    @abstractmethod
    def mutated(self, **attrs: Any) -> "Tensor": ...

    @abstractmethod
    def clone(self) -> "Tensor": ...

    @abstractmethod
    def contiguous(self) -> "Tensor": ...

    @abstractmethod
    def dot(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def square(self) -> "Tensor": ...

    @abstractmethod
    def sqrt(self) -> "Tensor": ...

    @abstractmethod
    def exp(self) -> "Tensor": ...

    @abstractmethod
    def log(self) -> "Tensor": ...

    @abstractmethod
    def sin(self) -> "Tensor": ...

    @abstractmethod
    def cos(self) -> "Tensor": ...

    @abstractmethod
    def sum(
        self, dim: Optional[Dimlike] = None, keepdims: bool = False
    ) -> "Tensor": ...

    @abstractmethod
    def max(
        self, dim: Optional[Dimlike] = None, keepdims: bool = False
    ) -> "Tensor": ...

    @abstractmethod
    def min(
        self, dim: Optional[Dimlike] = None, keepdims: bool = False
    ) -> "Tensor": ...

    @abstractmethod
    def squeeze(self, dim: Optional[Dimlike] = None) -> "Tensor": ...

    @abstractmethod
    def unsqueeze(self, dim: Dimlike) -> "Tensor": ...

    @abstractmethod
    def reshape(self, newdim: Dim) -> "Tensor": ...

    @abstractmethod
    def transpose(self, dim0: int = -2, dim1: int = -1) -> "Tensor": ...

    @abstractmethod
    def permute(self, dims: Dim) -> "Tensor": ...

    @abstractmethod
    def any(
        self, dim: Optional[Dimlike] = None, keepdims: bool = False
    ) -> "Tensor": ...

    @abstractmethod
    def all(
        self, dim: Optional[Dimlike] = None, keepdims: bool = False
    ) -> "Tensor": ...

    @abstractmethod
    def __add__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __radd__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __iadd__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __sub__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __rsub__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __isub__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __mul__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __rmul__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __imul__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __truediv__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __rtruediv__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __itruediv__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __floordiv__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __rfloordiv__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __ifloordiv__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __mod__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __rmod__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __imod__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __matmul__(self, other: "Tensor") -> "Tensor": ...

    @abstractmethod
    def __imatmul__(self, other: "Tensor") -> Self: ...

    @abstractmethod
    def __pow__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __rpow__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __ipow__(self, other: Union["Tensor", Scalar]) -> Self: ...

    @abstractmethod
    def __pos__(self) -> "Tensor": ...

    @abstractmethod
    def __neg__(self) -> "Tensor": ...

    @abstractmethod
    def __abs__(self) -> "Tensor": ...

    @abstractmethod
    def __invert__(self) -> "Tensor": ...

    @abstractmethod
    def __eq__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __lt__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __le__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __gt__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __ge__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __ne__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __hash__(self) -> int: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __bool__(self) -> None: ...

    @abstractmethod
    def __and__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __or__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def __xor__(self, other: Union["Tensor", Scalar]) -> "Tensor": ...

    @abstractmethod
    def to(self, dtype: Type[Dtype]) -> "Tensor": ...

    @abstractmethod
    def byte(self) -> "Tensor": ...

    @abstractmethod
    def char(self) -> "Tensor": ...

    @abstractmethod
    def short(self) -> "Tensor": ...

    @abstractmethod
    def int(self) -> "Tensor": ...

    @abstractmethod
    def long(self) -> "Tensor": ...

    @abstractmethod
    def half(self) -> "Tensor": ...

    @abstractmethod
    def float(self) -> "Tensor": ...

    @abstractmethod
    def double(self) -> "Tensor": ...

    @abstractmethod
    def bool(self) -> "Tensor": ...

    @abstractmethod
    def __del__(self): ...

    @abstractmethod
    def __setattr__(self, name: str, value: Any) -> None: ...

    @abstractmethod
    def __getitem__(self, slice_: Union[Tensorlike, "Tensor", slice]) -> "Tensor": ...

    @abstractmethod
    def __setitem__(self, slc: Any, item: Any) -> None: ...

    @abstractmethod
    def __repr__(self) -> str: ...


class Context(ABC):

    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def save(self, *tensors: Tensor) -> None: ...

    @abstractmethod
    def tensors(self) -> Tuple[Tensor, ...]: ...

    @abstractmethod
    def usesgrad(self) -> bool: ...

    @abstractmethod
    def __getattr__(self, name: str) -> Any: ...

    @abstractmethod
    def __setattr__(self, name: str, value: Any) -> None: ...

    @abstractmethod
    def __repr__(self) -> str: ...


class Function(ABC):

    @staticmethod
    @abstractmethod
    def forward(
        context: Context, *args: Any, **kwargs: Any
    ) -> Union[Tuple[ndarray, ...], ndarray]: ...

    @staticmethod
    @abstractmethod
    def backward(
        context: Context, grad: Tensor
    ) -> Union[Tuple[ndarray, ...], ndarray]: ...

    @staticmethod
    @abstractmethod
    def tangent(
        context: Context, *grad: Tensor
    ) -> Union[Tuple[ndarray, ...], ndarray]: ...

    @classmethod
    def apply(cls, *args: Any, **kwargs: Any) -> Any: ...

    @classmethod
    def name(cls) -> str: ...

import nura
import nura.types as types
from nura.types import Tensorlike, Scalar, dtype, dim, dimlike
from nura.autograd.graph import Node
from typing import Optional, Type, Any, Union, List, Self
from numpy import ndarray


class Tensor:

    def __init__(
        self,
        data: ndarray,
        usegrad: bool,
        grad: Optional["Tensor"],
        backfn: Optional[Node],
        leaf: bool,
    ) -> None:
        self._data: ndarray = data
        self._grad: Optional[Tensor] = grad
        self._backfn: Optional[Node] = backfn
        self._usegrad: bool = usegrad
        self._leaf: bool = leaf
        self._graph: int = 0

    @property
    def data(self) -> ndarray:
        return self._data

    @property
    def dim(self) -> dim:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def nelem(self) -> int:
        return self._data.size

    @property
    def usegrad(self) -> bool:
        return self._usegrad

    @property
    def grad(self) -> Optional["Tensor"]:
        return self._grad

    @property
    def backfn(self) -> Optional[Node]:
        return self._backfn

    @property
    def leaf(self) -> bool:
        return self._leaf

    @property
    def graph(self) -> int:
        return self._graph

    @property
    def dtype(self) -> Type[dtype]:
        return types.dtypeof(self.data)

    @property
    def gradtensor(self) -> bool:
        return self.dtype in (types.half, types.float, types.double)

    @property
    def T(self) -> "Tensor":
        revdims = tuple(range(self.ndim - 1, -1, -1))
        return self.permute(revdims)

    def item(self) -> Scalar:
        if self.nelem != 1:
            raise RuntimeError(
                f"Cannot retrieve a single element from a Tensor with {self.nelem} elements"
            )
        return self.data.item()

    def list(self) -> List[Any]:
        return self.data.tolist()

    def backward(self, grad: Optional["Tensor"] = None) -> None:
        nura.backward(self, grad)

    def cleargrad(self) -> None:
        self._grad = None

    def clearedgrad(self) -> "Tensor":
        cls = type(self)
        return cls(self.data, self.usegrad, None, self.backfn, self.leaf)

    def zerograd(self) -> None:
        self._grad = nura.zeroslike(self)

    def zeroedgrad(self) -> "Tensor":
        cls = type(self)
        return cls(self.data, self.usegrad, nura.zeroslike(self), None, True)

    def attach(self) -> None:
        self._usegrad = True

    def attached(self) -> "Tensor":
        cls = type(self)
        return cls(self.data, True, self.grad, self.backfn, self.leaf)

    def detach(self) -> None:
        self._usegrad = False

    def detached(self) -> "Tensor":
        return nura.detached(self)

    def mutate(self, **attrs: Any) -> None:
        for k, v in attrs.items():
            setattr(self, f"_{k}", v)

    def mutated(self, **attrs: Any) -> "Tensor":
        cls = type(self)
        t = cls(self.data, self.usegrad, self.grad, self.backfn, self.leaf)
        for k, v in attrs.items():
            setattr(t, f"_{k}", v)
        return t

    def clone(self) -> "Tensor":
        return nura.clone(self)

    def contiguous(self) -> "Tensor":
        return nura.tocontiguous(self)

    def dot(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.dot(self, other)

    def square(self) -> "Tensor":
        return nura.square(self)

    def sqrt(self) -> "Tensor":
        return nura.sqrt(self)

    def exp(self) -> "Tensor":
        return nura.exp(self)

    def log(self) -> "Tensor":
        return nura.log(self)

    def sin(self) -> "Tensor":
        return nura.sin(self)

    def cos(self) -> "Tensor":
        return nura.cos(self)

    def sum(self, dim: Optional[dimlike] = None, keepdims: bool = False) -> "Tensor":
        return nura.sum(self, dim, keepdims)

    def max(self, dim: Optional[dimlike] = None, keepdims: bool = False) -> "Tensor":
        return nura.max(self, dim, keepdims)

    def min(self, dim: Optional[dimlike] = None, keepdims: bool = False) -> "Tensor":
        return nura.min(self, dim, keepdims)

    def squeeze(self, dim: Optional[dimlike] = None) -> "Tensor":
        return nura.squeeze(self, dim)

    def unsqueeze(self, dim: dimlike) -> "Tensor":
        return nura.unsqueeze(self, dim)

    def view(self, newdim: types.dim) -> "Tensor":
        return nura.view(self, newdim)

    def reshape(self, newdim: types.dim) -> "Tensor":
        return nura.reshape(self, newdim)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> "Tensor":
        return nura.transpose(self, dim0, dim1)

    def permute(self, dims: types.dim) -> "Tensor":
        return nura.permute(self, dims)

    def any(self, dim: Optional[dimlike] = None, keepdims: bool = False) -> "Tensor":
        return nura.tensorany(self, dim, keepdims)

    def all(self, dim: Optional[dimlike] = None, keepdims: bool = False) -> "Tensor":
        return nura.tensorall(self, dim, keepdims)

    def __add__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.add(self, other)

    def __radd__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.add(self, other)

    def __iadd__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.iadd(self, other)
        return self

    def __sub__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.sub(self, other)

    def __rsub__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.sub(tensor(other, dtype=self.dtype), self)

    def __isub__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.isub(self, other)
        return self

    def __mul__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.mul(self, other)

    def __rmul__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.mul(self, other)

    def __imul__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.imul(self, other)
        return self

    def __truediv__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.div(self, other)

    def __rtruediv__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.div(tensor(other, dtype=self.dtype), self)

    def __itruediv__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.idiv(self, other)
        return self

    def __floordiv__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.floordiv(self, other)

    def __rfloordiv__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.floordiv(tensor(other, dtype=self.dtype), self)

    def __ifloordiv__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.ifloordiv(self, other)
        return self

    def __mod__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.modulo(self, other)

    def __rmod__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.modulo(tensor(other, dtype=self.dtype), self)

    def __imod__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.imodulo(self, other)
        return self

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return nura.matmul(self, other)

    def __imatmul__(self, other: "Tensor") -> Self:
        nura.imatmul(self, other)
        return self

    def __pow__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.pow(self, other)

    def __rpow__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.pow(tensor(other, dtype=self.dtype), self)

    def __ipow__(self, other: Union["Tensor", Scalar]) -> Self:
        nura.ipow(self, other)
        return self

    def __pos__(self) -> "Tensor":
        return nura.pos(self)

    def __neg__(self) -> "Tensor":
        return nura.neg(self)

    def __abs__(self) -> "Tensor":
        return nura.abs(self)

    def __invert__(self) -> "Tensor":
        return nura.tensornot(self)

    def __eq__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.equal(self, other)

    def __lt__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.less(self, other)

    def __le__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.lesseq(self, other)

    def __gt__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.greater(self, other)

    def __ge__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.greatereq(self, other)

    def __ne__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.notequal(self, other)

    def __hash__(self) -> int:
        return nura.hashtensor(self)

    def __len__(self) -> int:
        return len(self.data)

    def __bool__(self) -> None:
        raise ValueError(
            "Truth of Tensor is undefined for more than one element, use any() or all()"
        )

    def __and__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.tensorand(self, other)

    def __or__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.tensoror(self, other)

    def __xor__(self, other: Union["Tensor", Scalar]) -> "Tensor":
        return nura.tensorxor(self, other)

    def to(self, dtype: Type[types.dtype]) -> "Tensor":
        return nura.to(self, dtype)

    def byte(self) -> "Tensor":
        return self.to(types.byte)

    def char(self) -> "Tensor":
        return self.to(types.char)

    def short(self) -> "Tensor":
        return self.to(types.short)

    def int(self) -> "Tensor":
        return self.to(types.int)

    def long(self) -> "Tensor":
        return self.to(types.long)

    def half(self) -> "Tensor":
        return self.to(types.half)

    def float(self) -> "Tensor":
        return self.to(types.float)

    def double(self) -> "Tensor":
        return self.to(types.double)

    def bool(self) -> "Tensor":
        return self.to(types.bool)

    def __setattr__(self, name: str, value: Any) -> None:
        validattrs = ("_data", "_usegrad", "_grad", "_backfn", "_leaf", "_graph")
        if name not in validattrs:
            raise AttributeError(f"{name} cannot be assigned to {nura.typename(self)}")
        if name == "_usegrad":
            if value and self.dtype not in (types.half, types.float, types.double):
                raise ValueError(
                    f"Only floating-point Tensors can use gradient, received {dtype.name()}"
                )
        if name == "_data" and name in self.__dict__ and "_graph" in self.__dict__:
            if self._graph and nura.usegrad():
                raise ValueError(
                    "Cannot modify the data of a Tensor on the computational graph"
                )
        self.__dict__[name] = value

    def __getitem__(self, slc: Any) -> "Tensor":
        return nura.slice(self, slc)

    def __setitem__(self, slc: Any, item: Any) -> None:
        if isinstance(slc, tuple):
            slc = tuple(i.data if isinstance(i, Tensor) else i for i in slc)
        if isinstance(slc, Tensor):
            slc = slc.data
        if isinstance(item, Tensor):
            item = item.data
        self.data[slc] = item

    def __repr__(self) -> str:
        s = repr(self._data).replace("array(", "").replace(",", "").replace(")", "")
        if " dtype" in s:
            i = s.index(" dtype")
            s = s[:i]
        reprs = ["Tensor(", s]
        if self.backfn is not None:
            reprs.append(f" backfn={self.backfn.function.__name__}")
        reprs.append(f" dtype={self.dtype.name()})")
        return "".join(reprs)


def tensor(
    data: Union[Tensor, Tensorlike],
    usegrad: bool = False,
    dtype: Optional[Type[dtype]] = None,
) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
        dtype = data.dtype
    if dtype is None:
        dtype = nura.dtypeof(data)
    data = dtype.numpy(data)
    return Tensor(data, usegrad, None, None, True)

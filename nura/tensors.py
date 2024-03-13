import nura
import nura.types as types
from nura.types import dtype, dim, dimlike
from nura.autograd.graph import Node
from typing import Optional, Type, Any
from numpy import ndarray
from copy import deepcopy


class Tensor:

    _gradtensor: Optional[bool] = None

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
    def dtype(self) -> Type[dtype]:
        return types.dtypeof(self.data)

    @property
    def gradtensor(self) -> bool:
        return self.dtype in (types.half, types.float, types.double)

    @property
    def T(self):
        return self.transpose()

    def item(self):
        if self.nelem != 1:
            raise RuntimeError(
                f"Cannot retrieve a single element from a Tensor with {self.nelem} elements"
            )
        return self.data.item()

    def to(self, dtype: Type[types.dtype]):
        return nura.to(self, dtype)

    def byte(self):
        return self.to(types.byte)

    def char(self):
        return self.to(types.char)

    def short(self):
        return self.to(types.short)

    def int(self):
        return self.to(types.int)

    def long(self):
        return self.to(types.long)

    def half(self):
        return self.to(types.half)

    def float(self):
        return self.to(types.float)

    def double(self):
        return self.to(types.double)

    def bool(self):
        return self.to(types.bool)

    def backward(self, grad: Optional["Tensor"] = None):
        nura.backward(self, grad)

    def zerograd(self):
        return muttensor(self, grad=nura.zeroslike(self))

    def zeroedgrad(self):
        cls = type(self)
        return cls(self.data, self.usegrad, nura.zeroslike(self), None, True)

    def mutated(self, **attrs: Any) -> "Tensor":
        cls = type(self)
        a = cls(self.data, self.usegrad, self.grad, self.backfn, self.leaf)
        return muttensor(a, **attrs)

    def mutate(self, **attrs: Any) -> "Tensor":
        return muttensor(self, **attrs)

    def copy(self) -> "Tensor":
        cls = type(self)
        return cls(self.data.copy(), self.usegrad, None, None, True)

    def deepcopy(self) -> "Tensor":
        grad = self.grad.copy() if self.grad is not None else None
        backfn = deepcopy(self.backfn) if self.backfn is not None else None
        cls = type(self)
        return cls(self.data.copy(), self.usegrad, grad, backfn, self.leaf)

    def detach(self):
        return tensor(self.data, False, self.dtype)

    def clone(self):
        return nura.clone(self)

    def contig(self):
        return nura.tocontig(self)

    def sum(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.sum(self, dim, keepdims)

    def max(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.max(self, dim, keepdims)

    def min(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.min(self, dim, keepdims)

    def squeeze(self, dim: Optional[dimlike] = None):
        return nura.squeeze(self, dim)

    def unsqueeze(self, dim: Optional[dimlike] = None):
        return nura.unsqueeze(self, dim)

    def view(self, newdim: types.dim):
        return nura.view(self, newdim)

    def reshape(self, dim: types.dim):
        return nura.reshape(self, dim)

    def transpose(self, dim0=-2, dim1=-1):
        return nura.transpose(self, dim0, dim1)

    def permute(self, dims: Optional[types.dim] = None):
        return nura.permute(self, dims=dims)

    def any(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.tensorany(self, dim, keepdims)

    def all(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.tensorall(self, dim, keepdims)

    def __add__(self, other):
        return nura.add(self, other)

    def __radd__(self, other):
        return nura.add(self, other)

    def __sub__(self, other):
        return nura.sub(self, other)

    def __rsub__(self, other):
        return nura.sub(other, self)

    def __mul__(self, other):
        return nura.mul(self, other)

    def __rmul__(self, other):
        return nura.mul(self, other)

    def __truediv__(self, other):
        return nura.div(self, other)

    def __rtruediv__(self, other):
        return nura.div(other, self)

    def __matmul__(self, other):
        return nura.matmul(self, other)

    def __rmatmul__(self, other):
        return nura.matmul(other, self)

    def __pow__(self, other):
        return nura.pow(self, other)

    def __rpow__(self, other):
        return nura.pow(other, self)

    def __pos__(self):
        return nura.pos(self)

    def __neg__(self):
        return nura.neg(self)

    def __abs__(self):
        return nura.abs(self)

    def __eq__(self, other):
        return nura.equal(self, other)

    def __lt__(self, other):
        return nura.less(self, other)

    def __le__(self, other):
        return nura.lesseq(self, other)

    def __gt__(self, other):
        return nura.greater(self, other)

    def __ge__(self, other):
        return nura.greatereq(self, other)

    def __ne__(self, other):
        return nura.notequal(self, other)

    def __hash__(self):
        return nura.hashtensor(self)

    def __and__(self, other):
        return nura.tensorand(self, other)

    def __or__(self, other):
        return nura.tensoror(self, other)

    def __not__(self):
        return nura.tensornot(self)

    def __setattr__(self, name, value):
        validnames = {
            "_data",
            "_usegrad",
            "_grad",
            "_backfn",
            "_leaf",
        }
        if name not in validnames:
            raise AttributeError(f"{name} cannot be assigned to {nura.typename(self)}")
        self.__dict__[name] = value

    def __getitem__(self, slc):
        return nura.slice(self, slc)

    def __setitem__(self, slc, item):
        self.data[slc] = item.data if nura.istensor(item) else item

    def __len__(self):
        return self.dim[0]

    def __repr__(self) -> str:
        base = repr(self._data).replace("array(", "").replace(",", "")
        if " dtype" in base:
            i = base.index(" dtype")
            base = base[:i]
        strings = ["tensor(", base]
        if self.backfn is not None:
            strings.append(f" backfn={repr(self.backfn)}")
        strings.append(f" dtype={self.dtype.name()})")
        return "".join(strings)


def tensor(data: Any, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = nura.dtypeof(data)
    data = dtype.numpy(data)
    if usegrad and dtype not in (nura.half, nura.float, nura.double):
        raise ValueError(
            f"Only Tensors floating-point tensors can use gradient, received {dtype.name()}"
        )
    return Tensor(data, usegrad, None, None, True)


def muttensor(tensor: Tensor, **attrs: Any) -> Tensor:
    validattrs = {
        "data": "_data",
        "usegrad": "_usegrad",
        "grad": "_grad",
        "backfn": "_backfn",
        "leaf": "_leaf",
    }
    for name, val in attrs.items():
        if name not in validattrs:
            raise AttributeError(
                f"{name} is not a mutable member of {nura.typename(tensor)}"
            )
        setattr(tensor, validattrs[name], val)
    return tensor

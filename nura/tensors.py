import nura
import nura.types as types
from nura.types import Tensorlike, dimlike, dim, dtype
from nura.autograd.graph import Node
from typing import Optional, Type, Any, Union
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
        revdims = tuple(range(self.ndim - 1, -1, -1))
        return self.permute(revdims)

    def item(self):
        if self.nelem != 1:
            raise RuntimeError(
                f"Cannot retrieve a single element from a Tensor with {self.nelem} elements"
            )
        return self.data.item()

    def list(self):
        return self.data.tolist()

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

    def cleargrad(self):
        self._grad = None

    def clearedgrad(self):
        cls = type(self)
        return cls(self.data, self.usegrad, None, self.backfn, self.leaf)

    def zerograd(self):
        self._grad = nura.zeroslike(self)

    def zeroedgrad(self):
        cls = type(self)
        return cls(self.data, self.usegrad, nura.zeroslike(self), None, True)

    def usesgrad(self):
        self._usegrad = True

    def usedgrad(self):
        cls = type(self)
        return cls(self.data, True, self.grad, self.backfn, self.leaf)

    def mutate(self, **attrs: Any):
        for k, v in attrs.items():
            setattr(self, f"_{k}", v)

    def mutated(self, **attrs: Any):
        cls = type(self)
        t = cls(self.data, self.usegrad, self.grad, self.backfn, self.leaf)
        for k, v in attrs.items():
            setattr(t, f"_{k}", v)
        return t

    def nograd(self):
        self._usegrad = False

    def detach(self):
        return nura.detach(self)

    def clone(self):
        return nura.clone(self)

    def contiguous(self):
        return nura.tocontiguous(self)

    def exp(self):
        return nura.exp(self)

    def sum(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.sum(self, dim, keepdims)

    def max(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.max(self, dim, keepdims)

    def min(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.min(self, dim, keepdims)

    def squeeze(self, dim: Optional[dimlike] = None):
        return nura.squeeze(self, dim)

    def unsqueeze(self, dim: dimlike):
        return nura.unsqueeze(self, dim)

    def view(self, newdim: types.dim):
        return nura.view(self, newdim)

    def reshape(self, newdim: types.dim):
        return nura.reshape(self, newdim)

    def transpose(self, dim0=-2, dim1=-1):
        return nura.transpose(self, dim0, dim1)

    def permute(self, dims: types.dim):
        return nura.permute(self, dims)

    def any(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.tensorany(self, dim, keepdims)

    def all(self, dim: Optional[dimlike] = None, keepdims=False):
        return nura.tensorall(self, dim, keepdims)

    def __add__(self, other):
        return nura.add(self, other)

    def __radd__(self, other):
        return nura.add(self, other)

    def __iadd__(self, other):
        return nura.iadd(self, other)

    def __sub__(self, other):
        return nura.sub(self, other)

    def __rsub__(self, other):
        return nura.sub(tensor(other, dtype=self.dtype), self)

    def __isub__(self, other):
        return nura.isub(self, other)

    def __mul__(self, other):
        return nura.mul(self, other)

    def __rmul__(self, other):
        return nura.mul(self, other)

    def __imul__(self, other):
        return nura.imul(self, other)

    def __truediv__(self, other):
        return nura.div(self, other)

    def __rtruediv__(self, other):
        return nura.div(tensor(other, dtype=self.dtype), self)

    def __itruediv__(self, other):
        return nura.idiv(self, other)

    def __matmul__(self, other):
        return nura.matmul(self, other)

    def __imatmul__(self, other):
        return nura.imatmul(self, other)

    def __pow__(self, other):
        return nura.pow(self, other)

    def __ipow__(self, other):
        return nura.ipow(self, other)

    def __rpow__(self, other):
        return nura.pow(tensor(other, dtype=self.dtype), self)

    def __pos__(self):
        return nura.pos(self)

    def __neg__(self):
        return nura.neg(self)

    def __abs__(self):
        return nura.abs(self)

    def __invert__(self):
        return nura.tensornot(self)

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

    def __len__(self):
        return len(self.data)

    def __bool__(self):
        raise ValueError(
            "Truth of Tensor is undefined for more than one element, use .any() or .all()"
        )

    def __and__(self, other):
        return nura.tensorand(self, other)

    def __or__(self, other):
        return nura.tensoror(self, other)

    def __xor__(self, other):
        return nura.tensorxor(self, other)

    def __setattr__(self, name, value):
        validattrs = (
            "_data",
            "_usegrad",
            "_grad",
            "_backfn",
            "_leaf",
        )
        if name not in validattrs:
            raise AttributeError(f"{name} cannot be assigned to {nura.typename(self)}")
        gradtypes = (types.half, types.float, types.double)
        if name == "_usegrad" and value and self.dtype not in gradtypes:
            raise ValueError(
                f"Only floating-point Tensors can use gradient, received {dtype.name()}"
            )
        self.__dict__[name] = value

    def __getitem__(self, slc):
        return nura.slice(self, slc)

    def __setitem__(self, slc, item):
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
            reprs.append(f" backfn={self.backfn}")
        reprs.append(f" dtype={self.dtype.name()})")
        return "".join(reprs)


def tensor(
    data: Union[Tensor, Tensorlike], usegrad=False, dtype: Optional[Type[dtype]] = None
) -> Tensor:
    if isinstance(data, Tensor):
        data = data.data
        dtype = data.dtype
    if dtype is None:
        dtype = nura.dtypeof(data)
    data = dtype.numpy(data)
    return Tensor(data, usegrad, None, None, True)

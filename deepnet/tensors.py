import deepnet
from deepnet.types import dtype, _dim
from deepnet.autograd.graph import Node
from typing import Optional, Type, Any
from numpy import ndarray
from copy import deepcopy


class Tensor:

    _gradtensor = ...

    def __init__(
        self,
        data: ndarray,
        usegrad: bool,
        grad: "Tensor",
        backfn: Node,
        leaf: bool,
        _dtype: Type[dtype],
    ) -> None:

        self._data: ndarray = data
        self._grad: Optional[Tensor] = grad
        self._backfn: Optional[Node] = backfn
        self._usegrad: bool = usegrad
        self._leaf: bool = leaf
        self._dtype: Type[dtype] = _dtype

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def dim(self) -> _dim:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def nelem(self) -> int:
        return self._data.size

    @property
    def usegrad(self):
        return self._usegrad

    @property
    def grad(self):
        return self._grad

    @property
    def backfn(self):
        return self._backfn

    @property
    def leaf(self):
        return self._leaf

    @classmethod
    def gradtensor(cls):
        assert cls != Tensor
        return cls._gradtensor

    def item(self):
        assert self.nelem == 1
        return self.data.item()

    def to(self, dtype):
        return deepnet.to(self, dtype)

    def byte(self):
        return self.to(deepnet.byte)

    def char(self):
        return self.to(deepnet.char)

    def short(self):
        return self.to(deepnet.short)

    def int(self):
        return self.to(deepnet.int)

    def long(self):
        return self.to(deepnet.long)

    def half(self):
        return self.to(deepnet.half)

    def float(self):
        return self.to(deepnet.float)

    def double(self):
        return self.to(deepnet.double)

    def bool(self):
        return self.to(deepnet.bool)

    def backward(self, grad: Optional["Tensor"] = None):
        deepnet.backward(self, grad)

    def zerograd(self):
        muttensor(self, grad=deepnet.zeroslike(self))
        return self

    def zeroedgrad(self):
        cls = getcls(self)
        return cls(self.data, self.usegrad, deepnet.zeroslike(self), None, True)

    def mutated(self, **attrs: Any) -> "Tensor":
        cls = getcls(self.dtype)
        a = cls(self.data, self.usegrad, self.grad, self.backfn, self.leaf)
        return muttensor(a, **attrs)

    def mutate(self, **attrs: Any) -> "Tensor":
        return muttensor(self, **attrs)

    def copy(self) -> "Tensor":
        cls = getcls(self.dtype)
        return cls(self.data.copy(), self.usegrad, None, None, True)

    def deepcopy(self) -> "Tensor":
        grad = self.grad.copy() if self.grad is not None else None
        backfn = deepcopy(self.backfn) if self.backfn is not None else None
        cls = getcls(self.dtype)
        return cls(self.data.copy(), self.usegrad, grad, backfn, self.leaf)

    def detach(self):
        cls = getcls(self)
        return cls(self.data, False, None, None, True)

    def clone(self):
        return deepnet.clone(self)

    def contig(self):
        return deepnet.tocontig(self)

    def sum(self, dim: Optional[_dim] = None, keepdims=False):
        return deepnet.sum(self, dim, keepdims)

    def max(self, dim: Optional[_dim] = None, keepdims=False):
        return deepnet.max(self, dim, keepdims)

    def min(self, dim: Optional[_dim] = None, keepdims=False):
        return deepnet.min(self, dim, keepdims)

    def squeeze(self, dim: Optional[_dim] = None):
        return deepnet.squeeze(self, dim)

    def unsqueeze(self, dim: _dim):
        return deepnet.unsqueeze(self, dim)

    def view(self, dim: _dim):
        return deepnet.view(self, dim)

    def reshape(self, dim: _dim):
        return deepnet.reshape(self, dim)

    def transpose(self, dim0=-2, dim1=-1):
        return deepnet.transpose(self, dim0, dim1)

    def permute(self, dim: Optional[_dim] = None):
        return deepnet.permute(self, dim=dim)

    def any(self, dim: Optional[_dim] = None, keepdims=False):
        return deepnet.any(self, dim, keepdims)

    def all(self, dim: Optional[_dim] = None, keepdims=False):
        return deepnet.all(self, dim, keepdims)

    def __add__(self, other):
        return deepnet.add(self, other)

    def __radd__(self, other):
        return deepnet.add(self, other)

    def __sub__(self, other):
        return deepnet.sub(self, other)

    def __rsub__(self, other):
        return deepnet.sub(other, self)

    def __mul__(self, other):
        return deepnet.mul(self, other)

    def __rmul__(self, other):
        return deepnet.mul(self, other)

    def __truediv__(self, other):
        return deepnet.div(self, other)

    def __rtruediv__(self, other):
        return deepnet.div(other, self)

    def __matmul__(self, other):
        return deepnet.matmul(self, other)

    def __rmatmul__(self, other):
        return deepnet.matmul(other, self)

    def __pow__(self, other):
        return deepnet.pow(self, other)

    def __rpow__(self, other):
        return deepnet.pow(other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return deepnet.mul(self, -1.0)

    def __abs__(self):
        return deepnet.abs(self)

    def __eq__(self, other):
        return deepnet.equal(self, other)

    def __lt__(self, other):
        return deepnet.less(self, other)

    def __le__(self, other):
        return deepnet.lesseq(self, other)

    def __gt__(self, other):
        return deepnet.greater(self, other)

    def __ge__(self, other):
        return deepnet.greatereq(self, other)

    def __ne__(self, other):
        return deepnet.notequal(self, other)

    def __hash__(self):
        return deepnet.hashtensor(self)

    def __and__(self, other):
        return deepnet.tensorand(self, other)

    def __or__(self, other):
        return deepnet.tensoror(self, other)

    def __not__(self):
        return deepnet.tensornot(self)

    def __setattr__(self, name, value):
        validnames = {
            "_data",
            "_usegrad",
            "_grad",
            "_backfn",
            "_leaf",
            "_dtype",
            "_mutable",
        }
        if name not in validnames:
            raise AttributeError(
                f"{name} cannot be assigned to {deepnet.typename(self)}"
            )
        self.__dict__[name] = value

    def __getitem__(self, slc):
        return deepnet.slice(self, slc)

    def __setitem__(self, slc, item):
        self.data[slc] = item.data if deepnet.istensor(item) else item

    def __len__(self):
        return self.dim[0]

    def __repr__(self) -> str:
        base = repr(self._data).replace("array(", "").replace(",", "")
        if " dtype" in base:
            i = base.index(" dtype")
            base = base[:i]
        s = "tensor(" + base
        if self.backfn:
            s += " backfn=" + str(self.backfn)
        s += " dtype=" + self.dtype.name()
        s += ")"
        return s


class ByteTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.byte)


class CharTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.char)


class ShortTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.short)


class IntTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.int)


class LongTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.long)


class HalfTensor(Tensor):
    _gradtensor = True

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.half)


class FloatTensor(Tensor):
    _gradtensor = True

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.float)


class DoubleTensor(Tensor):
    _gradtensor = True

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.double)


class BoolTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, deepnet.bool)


def getcls(dtype) -> Type:
    dtypemap = {
        deepnet.byte: ByteTensor,
        deepnet.char: CharTensor,
        deepnet.short: ShortTensor,
        deepnet.int: IntTensor,
        deepnet.long: LongTensor,
        deepnet.half: HalfTensor,
        deepnet.float: FloatTensor,
        deepnet.double: DoubleTensor,
        deepnet.bool: BoolTensor,
    }
    return dtypemap[dtype]


def tensor(data: Any, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = deepnet.dtypeof(data)
    data = dtype.numpy(data)
    cls = getcls(dtype)
    if usegrad:
        assert cls.gradtensor(), f"{cls.__name__} cannot usegrad"
    return cls(data, usegrad, None, None, True)


def muttensor(tensor: Tensor, **attrs: Any) -> Tensor:
    validattrs = {
        "data": "_data",
        "usegrad": "_usegrad",
        "grad": "_grad",
        "backfn": "_backfn",
        "leaf": "_leaf",
        "dtype": "_dtype",
    }
    for name, val in attrs.items():
        if name in validattrs:
            setattr(tensor, validattrs[name], val)
    return tensor

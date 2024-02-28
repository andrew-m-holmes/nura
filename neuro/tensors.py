import neuro
from neuro.types import dtype, _dim
from neuro.autograd.graph import Node
from typing import Optional, Type, Any, Union
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
        return neuro.to(self, dtype)

    def byte(self):
        return self.to(neuro.byte)

    def char(self):
        return self.to(neuro.char)

    def short(self):
        return self.to(neuro.short)

    def int(self):
        return self.to(neuro.int)

    def long(self):
        return self.to(neuro.long)

    def half(self):
        return self.to(neuro.half)

    def float(self):
        return self.to(neuro.float)

    def double(self):
        return self.to(neuro.double)

    def bool(self):
        return self.to(neuro.bool)

    def backward(self, grad: Optional["Tensor"] = None):
        neuro.backward(self, grad)

    def zerograd(self):
        muttensor(self, grad=neuro.zeroslike(self))
        return self

    def zeroedgrad(self):
        cls = getcls(self.dtype)
        return cls(self.data, self.usegrad, neuro.zeroslike(self), None, True)

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
        return neuro.clone(self)

    def contig(self):
        return neuro.tocontig(self)

    def sum(self, dim: Optional[Union[_dim, int]] = None, keepdims=False):
        return neuro.sum(self, dim, keepdims)

    def max(self, dim: Optional[Union[_dim, int]] = None, keepdims=False):
        return neuro.max(self, dim, keepdims)

    def min(self, dim: Optional[Union[_dim, int]] = None, keepdims=False):
        return neuro.min(self, dim, keepdims)

    def squeeze(self, dim: Optional[_dim] = None):
        return neuro.squeeze(self, dim)

    def unsqueeze(self, dim: _dim):
        return neuro.unsqueeze(self, dim)

    def view(self, dim: _dim):
        return neuro.view(self, dim)

    def reshape(self, dim: _dim):
        return neuro.reshape(self, dim)

    def transpose(self, dim0=-2, dim1=-1):
        return neuro.transpose(self, dim0, dim1)

    def permute(self, dim: Optional[_dim] = None):
        return neuro.permute(self, dim=dim)

    def any(self, dim: Optional[Union[_dim, int]] = None, keepdims=False):
        return neuro.any(self, dim, keepdims)

    def all(self, dim: Optional[Union[_dim, int]] = None, keepdims=False):
        return neuro.all(self, dim, keepdims)

    def __add__(self, other):
        return neuro.add(self, other)

    def __radd__(self, other):
        return neuro.add(self, other)

    def __sub__(self, other):
        return neuro.sub(self, other)

    def __rsub__(self, other):
        return neuro.sub(other, self)

    def __mul__(self, other):
        return neuro.mul(self, other)

    def __rmul__(self, other):
        return neuro.mul(self, other)

    def __truediv__(self, other):
        return neuro.div(self, other)

    def __rtruediv__(self, other):
        return neuro.div(other, self)

    def __matmul__(self, other):
        return neuro.matmul(self, other)

    def __rmatmul__(self, other):
        return neuro.matmul(other, self)

    def __pow__(self, other):
        return neuro.pow(self, other)

    def __rpow__(self, other):
        return neuro.pow(other, self)

    def __pos__(self):
        return self

    def __neg__(self):
        return neuro.mul(self, -1.0)

    def __abs__(self):
        return neuro.abs(self)

    def __eq__(self, other):
        return neuro.equal(self, other)

    def __lt__(self, other):
        return neuro.less(self, other)

    def __le__(self, other):
        return neuro.lesseq(self, other)

    def __gt__(self, other):
        return neuro.greater(self, other)

    def __ge__(self, other):
        return neuro.greatereq(self, other)

    def __ne__(self, other):
        return neuro.notequal(self, other)

    def __hash__(self):
        return neuro.hashtensor(self)

    def __and__(self, other):
        return neuro.tensorand(self, other)

    def __or__(self, other):
        return neuro.tensoror(self, other)

    def __not__(self):
        return neuro.tensornot(self)

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
                f"{name} cannot be assigned to {neuro.typename(self)}"
            )
        self.__dict__[name] = value

    def __getitem__(self, slc):
        return neuro.slice(self, slc)

    def __setitem__(self, slc, item):
        self.data[slc] = item.data if neuro.istensor(item) else item

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
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.byte)


class CharTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.char)


class ShortTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.short)


class IntTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.int)


class LongTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.long)


class HalfTensor(Tensor):
    _gradtensor = True

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.half)


class FloatTensor(Tensor):
    _gradtensor = True

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.float)


class DoubleTensor(Tensor):
    _gradtensor = True

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.double)


class BoolTensor(Tensor):
    _gradtensor = False

    def __init__(self, data, usegrad, grad, backfn, leaf) -> None:
        super().__init__(data, usegrad, grad, backfn, leaf, neuro.bool)


def getcls(dtype) -> Type:
    dtypemap = {
        neuro.byte: ByteTensor,
        neuro.char: CharTensor,
        neuro.short: ShortTensor,
        neuro.int: IntTensor,
        neuro.long: LongTensor,
        neuro.half: HalfTensor,
        neuro.float: FloatTensor,
        neuro.double: DoubleTensor,
        neuro.bool: BoolTensor,
    }
    return dtypemap[dtype]


def tensor(data: Any, usegrad=False, dtype: Optional[Type[dtype]] = None) -> Tensor:
    if dtype is None:
        dtype = neuro.dtypeof(data)
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

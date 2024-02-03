import deepnet
from deepnet.dtype import dtype
from deepnet.graph import Node
from typing import Tuple, Optional
from numpy import ndarray


class TensorBase:

    def __init__(self, data, _dtype) -> None:
        self._data: ndarray = data
        self._dtype: dtype = _dtype

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def dim(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def nelem(self) -> int:
        return self._data.size

    def item(self):
        assert self.nelem == 1
        return self._data.item()

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

    def clone(self):
        return deepnet.clone(self)

    def contig(self):
        return deepnet.tocontig(self)

    def sum(self, dims=None, keepdims=False):
        return deepnet.sum(self, dims, keepdims)

    def squeeze(self, dims=None):
        return deepnet.squeeze(self, dims)

    def unsqueeze(self, dims):
        return deepnet.unsqueeze(self, dims)

    def view(self, dim):
        return deepnet.view(self, dim)

    def reshape(self, dim):
        return deepnet.reshape(self, dim)

    def transpose(self, dim_0=-2, dim_1=-1):
        return deepnet.transpose(self, dim_0, dim_1)

    def permute(self, dims=None):
        return deepnet.permute(self, dims=dims)

    def __add__(self, other):
        return deepnet.add(self, other)

    def __radd__(self, other):
        return deepnet.add(self, other)

    def __sub__(self, other):
        return deepnet.sub(self, other)

    def __pos__(self):
        raise NotImplementedError

    def __neg__(self):
        return deepnet.mul(self, -1.0)

    def __abs__(self):
        raise NotImplementedError

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

    def __getitem__(self, _slice):
        return deepnet.slice(self, _slice)

    def __setitem__(self, _slice, item):
        if issubclass(item, TensorBase):
            self.data[_slice] = item.data
        else:
            self.data[_slice] = item

    def __len__(self):
        return self._data.shape[0]

    def __repr__(self) -> str:
        return str(self._data)


class Tensor(TensorBase):

    def __init__(self, data, mut, grad, backfn, leaf, dtype) -> None:
        super().__init__(data, dtype)
        self._grad: Tensor = grad
        self._backfn: Optional[Node] = backfn
        self._mut: bool = mut
        self._leaf: bool = leaf

    @property
    def grad(self):
        return self._grad

    @property
    def backfn(self):
        return self._backfn

    @property
    def leaf(self):
        return self._leaf

    @property
    def mutable(self):
        return self._mut

    def backward(self, grad: Optional["Tensor"] = None):
        deepnet.backward(self, grad)

    def const(self):
        return tensor(self.data, False, self.dtype)

    def mut(self):
        return tensor(self.data, True, self.dtype)

    def mutated(
        self, data=None, mut=None, grad=None, backfn=None, leaf=True
    ) -> "Tensor":
        if data is None:
            data = self.data
        if mut is None:
            mut = self.mutable
        if grad is None:
            grad = self.grad
        if backfn is None:
            backfn = self.backfn
        return Tensor(data, mut, grad, backfn, leaf, self.dtype)

    def mutate(self, data=None, grad=None, backfn=None, leaf=True) -> "Tensor":
        assert self.mutable
        if data is not None:
            self._data = data
        if grad is not None:
            self._grad = grad
        if backfn is not None:
            self._backfn = backfn
        self._leaf = leaf
        return self

    def __repr__(self):
        base = super().__repr__()
        s = "tensor(" + base
        if self.backfn:
            s += " backfn=" + str(self.backfn)
        s += " dtype=" + self.dtype.name()
        s += ")"
        return s


def tensor(data, mut=False, dtype: Optional[dtype] = None) -> Tensor:
    if dtype is None:
        dtype = deepnet.dtypeof(data)
    if mut:
        assert dtype.candiff()
    data = dtype.numpy(data)
    return Tensor(data, mut, None, None, True, dtype)

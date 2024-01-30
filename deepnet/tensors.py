import deepnet
from typing import Tuple


class TensorBase:

    def __init__(self, data, dtype) -> None:
        self._data = data
        self._dtype = dtype

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

    def contiguous(self):
        return deepnet.to_contiguous(self)

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
        base = repr(self._data)
        rep = base.replace("array", "").replace(",", "").replace(")", "")
        if " dtype" in rep:
            start = rep.index(" dtype")
            rep = rep[:start]
        rep = "tensor(" + rep
        return rep + ")"


class Tensor(TensorBase):

    def __init__(self, data, diff, grad, backfn, leaf, dtype) -> None:
        super().__init__(data, dtype)
        self._diff = diff
        self._grad = grad
        self._backfn = backfn
        self._leaf = leaf

    @property
    def diff(self):
        return self._diff

    @property
    def grad(self):
        return self._grad

    @property
    def backfn(self):
        return self._backfn

    @property
    def leaf(self):
        return self._leaf
        
    def backward(self, grad=None):
        deepnet.backward(self, grad)

    def withstate(self, data=None, diff=None, grad=None, backfn=None, leaf=True):
        if data is None:
            data = self.data
        if diff is None:
            diff = self.diff
        if grad is None:
            grad = self.grad
        if backfn is None:
            backfn = self.backfn

        self._data = data
        self._diff = diff
        self._grad = grad
        self._backfn = backfn
        self._leaf = leaf
        return self


def tensor(data, diff=False, dtype=None):
    if dtype is None:
        dtype = deepnet.get_dtype(data)
    if diff:
        assert dtype.can_diff
    data = dtype.numpy(data)
    return Tensor(data, diff, None, None, True, dtype)

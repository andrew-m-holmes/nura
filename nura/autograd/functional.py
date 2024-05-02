import numpy as np
import nura
from nura.tensors import Tensor
from typing import Dict, Generator, Tuple, Optional, Callable, Any, Union, List
from collections import deque


def backward(out: Tensor, grad: Optional[Tensor] = None) -> None:
    raise NotImplemented


def _backward(
    out: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> None:
    raise NotImplemented


def _backwarderr(
    out: Tensor, grad: Optional[Tensor] = None
) -> Optional[Union[RuntimeError, ValueError]]:
    raise NotImplemented


def grad(
    out: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> Tuple[Tensor, ...]:
    raise NotImplemented


def _grad(
    inpt: Tuple[Tensor, ...], out: Tensor, grad: Optional[Tensor] = None
) -> Dict[Tensor, Tensor]:
    raise NotImplemented


def _graderr(
    inpt: Tuple[Tensor, ...], out: Tensor, grad: Optional[Tensor] = None
) -> Optional[Union[ValueError, RuntimeError]]:
    raise NotImplemented


def mapify(keys, values) -> Dict[Tensor, Any]:
    return {k: v for k, v in zip(keys, values)}


def tupify(inpt) -> Tuple[Tensor, ...]:
    if isinstance(inpt, Tensor):
        return (inpt,)
    return inpt


def vjp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:

    inpt = tupify(inpt)
    if err := _vjperr(inpt, vec):
        raise err
    inpt = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in inpt)
    vec = vec.mutated(usegrad=False, grad=None)
    out, grads = _vjp(inpt, vec, f, *args, **kwargs)
    return out.mutated(usegrad=False, gradfn=None, leaf=True), grads


def _vjp(
    inpt: Tuple[Tensor, ...],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    if err := _vjperr(inpt, vec):
        raise err
    with nura.autograd(enabled=True, reverse=True, forward=False):
        out = f(*inpt, *args, **kwargs)
    inptmap = _grad(inpt, out, vec)
    return out, tuple(inptmap.values())


def _vjperr(inpt: Tuple[Tensor, ...], vec: Tensor) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inpt):
        return ValueError(
            "One or more Tensors passed to argument 'inpt' cannot have their gradients computed because they're not differentiable types"
        )
    if not vec.gradtensor:
        return ValueError(
            f"Expected Tensor passed to 'vec' to be a floating-point type, received {vec.dtype.name()}"
        )
    return None


def jvp(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    vec: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inpt = tupify(inpt)
    vec = tupify(vec)
    if err := _jvperr(inpt, vec):
        raise err
    gen = (v for v in vec)
    inpt = tuple(t.mutated(usegrad=True, grad=next(gen)) for t in inpt)
    out, grad = _jvp(inpt, f, *args, **kwargs)
    return out.mutated(usegrad=False, leaf=True), grad


def _jvp(
    inpt: Tuple[Tensor, ...],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    with nura.autograd(enabled=True, reverse=False, forward=True):
        out = f(*inpt, *args, **kwargs)
    assert out.grad is not None
    return out, out.grad


def _jvperr(inpt: Tuple[Tensor, ...], vec: Tuple[Tensor, ...]) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inpt):
        return ValueError(
            "One or more Tensors passed to argument 'inpt' cannot have their gradients computed because they're not differentiable types"
        )
    if not all(v.gradtensor for v in vec):
        return ValueError(
            "One or more Tensors passed to argument 'vec' cannot be used to compute jvp() because they're not a floating-point type"
        )
    return None


def jacrev(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos=0,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inpt = tupify(inpt)
    if err := _jacerr(inpt):
        raise err
    inpt = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in inpt)
    with nura.autograd(enabled=True, reverse=True, forward=False):
        out = f(*inpt, *args, **kwargs)
    tensor = inpt[pos]
    jac = getjac(tensor, out)
    perts = getperts(out)

    for row, pert in zip(np.ndindex(out.dim), perts):
        _, grads = _vjp(inpt, pert, f, *args, **kwargs)
        jacrow = grads[pos]
        slc = row + (...,)
        jac[slc] = jacrow
    return out, jac


def jacfwd(
    inpt: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos=0,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inpt = tupify(inpt)
    if err := _jacerr(inpt):
        raise err
    with nura.autograd(enabled=False):
        out = f(*inpt, *args, **kwargs)
    tensor = inpt[pos]
    perts = getperts(tensor)
    jac = getjac(tensor, out)
    colinpt = [
        t.mutated(usegrad=True, grad=nura.zeroslike(t)) if i != pos else t
        for i, t in enumerate(inpt)
    ]
    for col, pert in zip(np.ndindex(tensor.dim), perts):
        colinpt[pos] = colinpt[pos].mutated(usegrad=True, grad=pert)
        _, jaccol = _jvp(tuple(colinpt), f, *args, **kwargs)
        slc = (...,) + col
        jac[slc] = jaccol
    return out, jac


def _jacerr(inpt: Tuple[Tensor, ...]) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inpt):
        return ValueError(
            "Cannot compute Jacobian because one or more Tensors passed to 'inpt' are not a floating-point type"
        )
    return None


def getperts(tensor: Tensor) -> Generator[Tensor, None, None]:
    nelem, dim, dtype = tensor.nelem, tensor.dim, tensor.dtype
    perts = nura.zeros((nelem,) + dim).to(dtype)
    arange = np.arange(nelem)
    indices = np.unravel_index(arange, dim)
    slc = (arange,) + indices
    perts[slc] = 1.0
    return (perts[i] for i in range(nelem))


def getjac(tensor: Tensor, out: Tensor) -> Tensor:
    dim = out.dim + tensor.dim
    jac = nura.zeros(dim).to(out.dtype)
    return jac

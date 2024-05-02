import numpy as np
import nura
from nura.tensors import Tensor
from nura.autograd.graph import Node
from typing import Dict, Generator, Tuple, Optional, Callable, Any, Union, List, Deque
from collections import deque


def backward(
    outputs: Union[Tuple[Tensor, ...], Tensor],
    grads: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> None:
    outputs, grads, inputs = tupify(outputs), tupify(grads), tupify(inputs)
    _backward(outputs, grads, inputs)


def _backward(
    outputs: Tuple[Tensor, ...],
    grads: Tuple[Tensor, ...],
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]],
) -> None:
    _grads = _makegrads(outputs, grads)
    queue = deque([(o.gradfn, g) for o, g in zip(outputs, _grads)])
    nodemap = dict((o.gradfn, o.gradfn.outputs) for o in outputs)

    while queue:
        node, grads = queue.popleft()
        pass


def _backwarderr(
    outputs: Union[Tuple[Tensor, ...], Tensor],
    grads: Optional[Union[Tuple[Tensor, ...], Tensor]],
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]],
) -> Optional[Union[RuntimeError, ValueError]]:
    pass


def grad(
    outputs: Union[Tuple[Tensor, ...], Tensor],
    grads: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> Tuple[Tensor, ...]:
    raise NotImplemented


def _grad(
    inputs: Tuple[Tensor, ...], outputs: Tensor, grads: Optional[Tensor] = None
) -> Dict[Tensor, Tensor]:
    raise NotImplemented


def _graderr(
    inputs: Tuple[Tensor, ...], outputs: Tensor, grads: Optional[Tensor] = None
) -> Optional[Union[ValueError, RuntimeError]]:
    raise NotImplemented


def _makegrads(
    outputs: Tuple[Tensor, ...],
    grads: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], ...]:
    _grads = []
    for i, o in enumerate(outputs):
        g = grads[i] if i < len(grads) else nura.ones()
        ograds = _nodegrads(o, g)
        _grads.append(ograds)
    return tuple(_grads)


def _nodegrads(tensor: Tensor, grad: Tensor) -> Tuple[Tensor, ...]:
    if tensor.gradfn is None:
        return ()
    return tuple(
        [
            grad if tensor.index == i else nura.zeros()
            for i in range(tensor.gradfn.outputs)
        ]
    )


def mapify(keys, values) -> Dict[Tensor, Any]:
    return {k: v for k, v in zip(keys, values)}


def tupify(inputs: Optional[Union[Tuple[Tensor, ...], Tensor]]) -> Tuple[Tensor, ...]:
    if inputs is None:
        return ()
    if isinstance(inputs, Tensor):
        return (inputs,)
    return inputs


def vjp(
    inputs: Union[Tuple[Tensor, ...], Tensor],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:

    inputs = tupify(inputs)
    if err := _vjperr(inputs, vec):
        raise err
    inputs = tuple(t.mutated(usegrads=True, grads=None, leaf=True) for t in inputs)
    vec = vec.mutated(usegrads=False, grads=None)
    outputs, gradss = _vjp(inputs, vec, f, *args, **kwargs)
    return outputs.mutated(usegrads=False, gradsfn=None, leaf=True), gradss


def _vjp(
    inputs: Tuple[Tensor, ...],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    if err := _vjperr(inputs, vec):
        raise err
    with nura.autograd(enabled=True, reverse=True, forward=False):
        outputs = f(*inputs, *args, **kwargs)
    inputsmap = _grads(inputs, outputs, vec)
    return outputs, tuple(inputsmap.values())


def _vjperr(inputs: Tuple[Tensor, ...], vec: Tensor) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inputs):
        return ValueError(
            "One or more Tensors passed to argument 'inputs' cannot have their gradsients computed because they're not differentiable types"
        )
    if not vec.gradtensor:
        return ValueError(
            f"Expected Tensor passed to 'vec' to be a floating-point type, received {vec.dtype.name()}"
        )
    return None


def jvp(
    inputs: Union[Tuple[Tensor, ...], Tensor],
    vec: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inputs = tupify(inputs)
    vec = tupify(vec)
    if err := _jvperr(inputs, vec):
        raise err
    gen = (v for v in vec)
    inputs = tuple(t.mutated(usegrads=True, grads=next(gen)) for t in inputs)
    outputs, grads = _jvp(inputs, f, *args, **kwargs)
    return outputs.mutated(usegrads=False, leaf=True), grads


def _jvp(
    inputs: Tuple[Tensor, ...],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    with nura.autograd(enabled=True, reverse=False, forward=True):
        outputs = f(*inputs, *args, **kwargs)
    assert outputs.grad is not None
    return outputs, outputs.grad


def _jvperr(
    inputs: Tuple[Tensor, ...], vec: Tuple[Tensor, ...]
) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inputs):
        return ValueError(
            "One or more Tensors passed to argument 'inputs' cannot have their gradsients computed because they're not differentiable types"
        )
    if not all(v.gradtensor for v in vec):
        return ValueError(
            "One or more Tensors passed to argument 'vec' cannot be used to compute jvp() because they're not a floating-point type"
        )
    return None


def jacrev(
    inputs: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos=0,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inputs = tupify(inputs)
    if err := _jacerr(inputs):
        raise err
    inputs = tuple(t.mutated(usegrads=True, grads=None, leaf=True) for t in inputs)
    with nura.autograd(enabled=True, reverse=True, forward=False):
        outputs = f(*inputs, *args, **kwargs)
    tensor = inputs[pos]
    jac = getjac(tensor, outputs)
    perts = getperts(outputs)

    for row, pert in zip(np.ndindex(outputs.dim), perts):
        _, gradss = _vjp(inputs, pert, f, *args, **kwargs)
        jacrow = gradss[pos]
        slc = row + (...,)
        jac[slc] = jacrow
    return outputs, jac


def jacfwd(
    inputs: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos=0,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    inputs = tupify(inputs)
    if err := _jacerr(inputs):
        raise err
    with nura.autograd(enabled=False):
        outputs = f(*inputs, *args, **kwargs)
    tensor = inputs[pos]
    perts = getperts(tensor)
    jac = getjac(tensor, outputs)
    colinputs = [
        t.mutated(usegrads=True, grads=nura.zeroslike(t)) if i != pos else t
        for i, t in enumerate(inputs)
    ]
    for col, pert in zip(np.ndindex(tensor.dim), perts):
        colinputs[pos] = colinputs[pos].mutated(usegrads=True, grads=pert)
        _, jaccol = _jvp(tuple(colinputs), f, *args, **kwargs)
        slc = (...,) + col
        jac[slc] = jaccol
    return outputs, jac


def _jacerr(inputs: Tuple[Tensor, ...]) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inputs):
        return ValueError(
            "Cannot compute Jacobian because one or more Tensors passed to 'inputs' are not a floating-point type"
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


def getjac(tensor: Tensor, outputs: Tensor) -> Tensor:
    dim = outputs.dim + tensor.dim
    jac = nura.zeros(dim).to(outputs.dtype)
    return jac

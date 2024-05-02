import numpy as np
import nura
from nura.tensors import Tensor
from nura.autograd.graph import Node
from typing import Dict, Generator, Tuple, Optional, Callable, Any, Union, List, Deque
from collections import deque


def backward(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> None:
    output, grad, inputs = tupify(output), tupify(grad), tupify(inputs)
    _backward(output, grad, inputs)


def _backward(
    output: Tuple[Tensor, ...],
    grad: Tuple[Tensor, ...],
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]],
) -> None:
    _grad = _makegrad(output, grad)
    print(_grad)


def _backwarderr(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]],
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]],
) -> Optional[Union[RuntimeError, ValueError]]:
    pass


def grad(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    inputs: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> Tuple[Tensor, ...]:
    raise NotImplemented


def _grad(
    inputs: Tuple[Tensor, ...], output: Tensor, grad: Optional[Tensor] = None
) -> Dict[Tensor, Tensor]:
    raise NotImplemented


def _graderr(
    inputs: Tuple[Tensor, ...], output: Tensor, grad: Optional[Tensor] = None
) -> Optional[Union[ValueError, RuntimeError]]:
    raise NotImplemented


def _makegrad(
    output: Tuple[Tensor, ...],
    grad: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], ...]:
    _grad = []
    for i, o in enumerate(output):
        g = grad[i] if i < len(grad) else nura.ones()
        ograd = _nodegrad(o, g)
        _grad.append(ograd)
    return tuple(_grad)


def _nodegrad(tensor: Tensor, grad: Tensor) -> Tuple[Tensor, ...]:
    if tensor.gradfn is None:
        return ()
    return tuple(
        grad if tensor.index == i else nura.zeros()
        for i in range(tensor.gradfn.outputs)
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
    inputs = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in inputs)
    vec = vec.mutated(usegrad=False, grad=None)
    output, grads = _vjp(inputs, vec, f, *args, **kwargs)
    return output.mutated(usegrad=False, gradfn=None, leaf=True), grads


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
        output = f(*inputs, *args, **kwargs)
    inputsmap = _grad(inputs, output, vec)
    return output, tuple(inputsmap.values())


def _vjperr(inputs: Tuple[Tensor, ...], vec: Tensor) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inputs):
        return ValueError(
            "One or more Tensors passed to argument 'inputs' cannot have their gradients computed because they're not differentiable types"
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
    inputs = tuple(t.mutated(usegrad=True, grad=next(gen)) for t in inputs)
    output, grad = _jvp(inputs, f, *args, **kwargs)
    return output.mutated(usegrad=False, leaf=True), grad


def _jvp(
    inputs: Tuple[Tensor, ...],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    with nura.autograd(enabled=True, reverse=False, forward=True):
        output = f(*inputs, *args, **kwargs)
    assert output.grad is not None
    return output, output.grad


def _jvperr(
    inputs: Tuple[Tensor, ...], vec: Tuple[Tensor, ...]
) -> Optional[ValueError]:
    if not all(t.gradtensor for t in inputs):
        return ValueError(
            "One or more Tensors passed to argument 'inputs' cannot have their gradients computed because they're not differentiable types"
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
    inputs = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in inputs)
    with nura.autograd(enabled=True, reverse=True, forward=False):
        output = f(*inputs, *args, **kwargs)
    tensor = inputs[pos]
    jac = getjac(tensor, output)
    perts = getperts(output)

    for row, pert in zip(np.ndindex(output.dim), perts):
        _, grads = _vjp(inputs, pert, f, *args, **kwargs)
        jacrow = grads[pos]
        slc = row + (...,)
        jac[slc] = jacrow
    return output, jac


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
        output = f(*inputs, *args, **kwargs)
    tensor = inputs[pos]
    perts = getperts(tensor)
    jac = getjac(tensor, output)
    colinputs = [
        t.mutated(usegrad=True, grad=nura.zeroslike(t)) if i != pos else t
        for i, t in enumerate(inputs)
    ]
    for col, pert in zip(np.ndindex(tensor.dim), perts):
        colinputs[pos] = colinputs[pos].mutated(usegrad=True, grad=pert)
        _, jaccol = _jvp(tuple(colinputs), f, *args, **kwargs)
        slc = (...,) + col
        jac[slc] = jaccol
    return output, jac


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


def getjac(tensor: Tensor, output: Tensor) -> Tensor:
    dim = output.dim + tensor.dim
    jac = nura.zeros(dim).to(output.dtype)
    return jac

import numpy as np
import nura
from numpy import ndarray
from nura.tensors import Tensor
from nura.autograd.graph import Node, constructgraph, topological
from typing import Dict, Generator, Tuple, Optional, Callable, Any, Union, List
from collections import deque, defaultdict


def backward(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    input: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> None:
    output, grad, input = tupify(output), tupify(grad), tupify(input)
    _backward(output, grad, input)


def _backward(
    output: Tuple[Tensor, ...],
    grad: Tuple[Tensor, ...],
    input: Optional[Union[Tuple[Tensor, ...], Tensor]],
) -> None:
    _grad = _makegrad(output, grad)
    nodes = tuple(o.gradfn for o in output if o.gradfn is not None)
    graph = constructgraph(nodes)
    order = topological(graph)
    nodegrads = _makenodegrads(order, nodes, _grad)
    queue = deque(order)

    # TODO handle accumulate
    while queue:
        node = queue.popleft()
        outputgrad = nodegrads[node]
        edgearr = node.apply(*outputgrad)
        nodegrads.pop(node)
        if node.edges is None:
            continue
        edgegrad = _postapply(edgearr)
        for (child, index), childgrad in zip(node.edges, edgegrad):
            if child is not None:
                nodegrads[child][index] = childgrad


def _backwarderr(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]],
    input: Optional[Union[Tuple[Tensor, ...], Tensor]],
) -> Optional[Union[RuntimeError, ValueError]]:
    pass


def grad(
    output: Union[Tuple[Tensor, ...], Tensor],
    grad: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
    input: Optional[Union[Tuple[Tensor, ...], Tensor]] = None,
) -> Tuple[Tensor, ...]:
    raise NotImplemented


def _grad(
    input: Tuple[Tensor, ...], output: Tensor, grad: Optional[Tensor] = None
) -> Dict[Tensor, Tensor]:
    raise NotImplemented


def _graderr(
    input: Tuple[Tensor, ...], output: Tensor, grad: Optional[Tensor] = None
) -> Optional[Union[ValueError, RuntimeError]]:
    raise NotImplemented


def _postapply(edgegrad: Union[Tuple[ndarray, ...], ndarray]) -> Tuple[Tensor, ...]:
    if isinstance(edgegrad, tuple):
        return tuple(nura.tensor(g) for g in edgegrad)
    return (nura.tensor(edgegrad),)


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


def _makenodegrads(
    toposort: List[Node], nodes: Tuple[Node, ...], grad: Tuple[Tuple[Tensor, ...], ...]
) -> Dict[Node, List[Optional[Tensor]]]:
    nodegrads: Dict[Node, List[Optional[Tensor]]] = {
        n: [None] * n.outputs for n in toposort
    }
    for n, g in zip(nodes, grad):
        nodegrads[n] = list(g)
    return nodegrads


def _nodegrad(tensor: Tensor, grad: Tensor) -> Tuple[Tensor, ...]:
    if tensor.gradfn is None:
        return ()
    return tuple(
        grad if tensor.index == i else nura.zeros()
        for i in range(tensor.gradfn.outputs)
    )


def mapify(keys, values) -> Dict[Tensor, Any]:
    return {k: v for k, v in zip(keys, values)}


def tupify(input: Optional[Union[Tuple[Tensor, ...], Tensor]]) -> Tuple[Tensor, ...]:
    if input is None:
        return ()
    if isinstance(input, Tensor):
        return (input,)
    return input


def vjp(
    input: Union[Tuple[Tensor, ...], Tensor],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:

    input = tupify(input)
    if err := _vjperr(input, vec):
        raise err
    input = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in input)
    vec = vec.mutated(usegrad=False, grad=None)
    output, grads = _vjp(input, vec, f, *args, **kwargs)
    return output.mutated(usegrad=False, gradfn=None, leaf=True), grads


def _vjp(
    input: Tuple[Tensor, ...],
    vec: Tensor,
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    if err := _vjperr(input, vec):
        raise err
    with nura.autograd(enabled=True, reverse=True, forward=False):
        output = f(*input, *args, **kwargs)
    inputmap = _grad(input, output, vec)
    return output, tuple(inputmap.values())


def _vjperr(input: Tuple[Tensor, ...], vec: Tensor) -> Optional[ValueError]:
    if not all(t.gradtensor for t in input):
        return ValueError(
            "One or more Tensors passed to argument 'input' cannot have their gradients computed because they're not differentiable types"
        )
    if not vec.gradtensor:
        return ValueError(
            f"Expected Tensor passed to 'vec' to be a floating-point type, received {vec.dtype.name()}"
        )
    return None


def jvp(
    input: Union[Tuple[Tensor, ...], Tensor],
    vec: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    input = tupify(input)
    vec = tupify(vec)
    if err := _jvperr(input, vec):
        raise err
    gen = (v for v in vec)
    input = tuple(t.mutated(usegrad=True, grad=next(gen)) for t in input)
    output, grad = _jvp(input, f, *args, **kwargs)
    return output.mutated(usegrad=False, leaf=True), grad


def _jvp(
    input: Tuple[Tensor, ...],
    f: Callable[..., Tensor],
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    with nura.autograd(enabled=True, reverse=False, forward=True):
        output = f(*input, *args, **kwargs)
    assert output.grad is not None
    return output, output.grad


def _jvperr(input: Tuple[Tensor, ...], vec: Tuple[Tensor, ...]) -> Optional[ValueError]:
    if not all(t.gradtensor for t in input):
        return ValueError(
            "One or more Tensors passed to argument 'input' cannot have their gradients computed because they're not differentiable types"
        )
    if not all(v.gradtensor for v in vec):
        return ValueError(
            "One or more Tensors passed to argument 'vec' cannot be used to compute jvp() because they're not a floating-point type"
        )
    return None


def jacrev(
    input: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos=0,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    input = tupify(input)
    if err := _jacerr(input):
        raise err
    input = tuple(t.mutated(usegrad=True, grad=None, leaf=True) for t in input)
    with nura.autograd(enabled=True, reverse=True, forward=False):
        output = f(*input, *args, **kwargs)
    tensor = input[pos]
    jac = getjac(tensor, output)
    perts = getperts(output)

    for row, pert in zip(np.ndindex(output.dim), perts):
        _, grads = _vjp(input, pert, f, *args, **kwargs)
        jacrow = grads[pos]
        slc = row + (...,)
        jac[slc] = jacrow
    return output, jac


def jacfwd(
    input: Union[Tuple[Tensor, ...], Tensor],
    f: Callable[..., Tensor],
    pos=0,
    *args,
    **kwargs,
) -> Tuple[Tensor, Tensor]:

    input = tupify(input)
    if err := _jacerr(input):
        raise err
    with nura.autograd(enabled=False):
        output = f(*input, *args, **kwargs)
    tensor = input[pos]
    perts = getperts(tensor)
    jac = getjac(tensor, output)
    colinput = [
        t.mutated(usegrad=True, grad=nura.zeroslike(t)) if i != pos else t
        for i, t in enumerate(input)
    ]
    for col, pert in zip(np.ndindex(tensor.dim), perts):
        colinput[pos] = colinput[pos].mutated(usegrad=True, grad=pert)
        _, jaccol = _jvp(tuple(colinput), f, *args, **kwargs)
        slc = (...,) + col
        jac[slc] = jaccol
    return output, jac


def _jacerr(input: Tuple[Tensor, ...]) -> Optional[ValueError]:
    if not all(t.gradtensor for t in input):
        return ValueError(
            "Cannot compute Jacobian because one or more Tensors passed to 'input' are not a floating-point type"
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

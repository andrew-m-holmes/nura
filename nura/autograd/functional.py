import numpy as np
import nura
from numpy import ndarray
from nura.tensors import Tensor
from nura.autograd.graph import Node, constructgraph, topological
from typing import Dict, Generator, Tuple, Optional, Callable, Union, List, Set
from collections import deque


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
    input: Tuple[Tensor, ...],
) -> None:
    nodes = tuple(o.gradfn for o in output if o.gradfn is not None)
    _grad = _make_grads(output, grad)
    graph = constructgraph(nodes)
    order = topological(graph)
    accumulates = _make_accumulation_set(graph, input)
    grad_map = _make_grad_map(order, nodes, _grad)
    queue = deque(order)

    while queue:
        node = queue.popleft()
        outputgrad = grad_map[node]
        if _retains_grad(node, accumulates):
            _accumulate_grad(node, outputgrad)
        if node.edges:
            edgearr = node.apply(*outputgrad)
            edgegrad = _postapply(edgearr)
            _backprop(node.edges, edgegrad, grad_map)
        grad_map.pop(node)


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


def _retains_grad(node: Node, accumulates: Set[Tensor]) -> bool:
    if node._tensors is None:
        return False
    return bool(len(node._tensors.intersection(accumulates)))


def _accumulate_grad(node: Node, output_grad: List[Optional[Tensor]]) -> None:
    if node._tensors is None:
        raise RuntimeError("Cannot accumulate gradient, Node has no tensors")
    if is_leaf(node):
        tensor, *_ = node._tensors
        if tensor.grad is None:
            tensor.zerograd()
        for grad in output_grad:
            tensor._grad += grad
    else:
        for tensor in node._tensors:
            if output_grad[tensor.index] is not None:
                tensor._grad += output_grad[tensor.index]


def _make_accumulation_set(
    graph: Dict[Node, List[Node]], input: Tuple[Tensor, ...]
) -> Set[Tensor]:
    accumulates = set(input)
    for node in graph.keys():
        if node._tensors is not None:
            accumulates.update(node._tensors)
    return accumulates


def _backprop(
    edges: Tuple[Tuple[Node, int], ...],
    edgegrad: Tuple[Tensor, ...],
    grad_map: Dict[Node, List[Optional[Tensor]]],
) -> None:
    for (node, index), grad in zip(edges, edgegrad):
        if node is None:
            continue
        if is_leaf(node):
            grad_map[node].append(grad)
        else:
            grad_map[node][index] = grad


def is_leaf(node: Node) -> bool:
    if node._tensors is None or len(node._tensors) > 1:
        return False
    tensor, *_ = node._tensors
    return tensor.leaf


def _make_grads(
    output: Tuple[Tensor, ...],
    grad: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], ...]:
    _grad = []
    for i, o in enumerate(output):
        g = grad[i] if i < len(grad) else nura.ones()
        ograd = _nodegrad(o, g)
        _grad.append(ograd)
    return tuple(_grad)


def _postapply(edgegrad: Union[Tuple[ndarray, ...], ndarray]) -> Tuple[Tensor, ...]:
    if isinstance(edgegrad, tuple):
        return tuple(nura.tensor(g) for g in edgegrad)
    return (nura.tensor(edgegrad),)


def _nodegrad(tensor: Tensor, grad: Tensor) -> Tuple[Tensor, ...]:
    assert tensor.gradfn is not None
    return tuple(
        grad if tensor.index == i else nura.zeros()
        for i in range(tensor.gradfn.outputs)
    )


def _make_grad_map(
    toposort: List[Node], nodes: Tuple[Node, ...], grad: Tuple[Tuple[Tensor, ...], ...]
) -> Dict[Node, List[Optional[Tensor]]]:
    grad_map: Dict[Node, List[Optional[Tensor]]] = {
        n: [None] * n.outputs for n in toposort
    }
    for n, g in zip(nodes, grad):
        grad_map[n] = list(g)
    return grad_map


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
            "One or more Tensors passed to argument 'input' cannot have their grads computed because they're not differentiable types"
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
            "One or more Tensors passed to argument 'input' cannot have their grads computed because they're not differentiable types"
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

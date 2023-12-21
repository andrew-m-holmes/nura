import deepnet
from types import FunctionType

def jacobian(input, func):
    # literally computes the full jacobian matrix for func
    # which holds the partial derivatives of the func outputs wrt
    # to every single input to that func
    pass


def vjp(primals, cotangent, func, use_graph=False):
    _vjp_args_check(primals, cotangent, use_graph)
    primals, cotangent = _vjp_pre_process(primals, cotangent, func, use_graph)
    with deepnet.use_grad():
        output = func(*primals)

    cotangents = []
    stack = [(output.grad_fn, cotangent)]
    while stack:
        node, cotangent = stack.pop()
        if _is_leaf_node(node) and node.context.tensor in primals:
            cotangents.append(cotangent)
        elif _is_intermediate_node(node):
            next_nodes, next_cotangents = _process_node(
                node, cotangent)
            for node, cotangent in zip(next_nodes, next_cotangents):
                if _is_leaf_node(node) or _is_intermediate_node(node):
                    stack.append((node, cotangent))
    return _vjp_post_process(output, cotangents, use_graph)


def _process_node(node, cotangent):
    next_cotangents = node.context.apply(cotangent)
    return node.next_functions, next_cotangents


def _vjp_post_process(output, cotangents, use_graph):
    if not use_graph:
        output._set_grad_state(
            use_grad=False, grad_fn=None, is_leaf=True)
    return output, tuple(reversed(cotangents))


def _vjp_pre_process(primals, cotangent, use_graph):
    tmp = primals
    primals = []
    for primal in tmp:
        if not use_graph:
            primal = primal.clone().detach()
        primal._set_grad_state(
            use_grad=True, grad_fn=None, is_leaf=True)
        primals.append(primal)
    cotangent._set_grad_state(
        use_grad=True, grad_fn=None, is_leaf=True)
    return primals, cotangent


def _vjp_args_check(primals, cotangent, func, use_graph):
    assert deepnet.is_all_tensor(*primals)
    assert deepnet.is_tensor(cotangent)
    assert isinstance(func, FunctionType)
    assert deepnet.is_py_bool(use_graph)
    assert all(tensor.dtype.differentiable() for tensor in primals)
    assert cotangent.dtype.differentiable()


def _is_leaf_node(node):
    if node is not None:
        return repr(node) == "AccumulateGrad"
    return False


def _is_intermediate_node(node):
    if node is not None:
        return "Backward" in repr(node)
    return False


def jvp(primals, tangents, func, use_graph=False):
    _jvp_args_check(primals, tangents, func, use_graph)
    primals = _jvp_pre_process_primals(primals, tangents, use_graph)

    with deepnet.forward_ad(), deepnet.set_grad(use_graph):
        output = func(*primals)
    output, tangent = _jvp_post_process(primals, output)
    return output, tangent
    

def _jvp_post_process(primals, output):
    for primal in primals:
        tensor, tangent = primal.undual(inplace=True)
    tangent = output.tangent
    output._set_dual_state(None, False)
    return output, tangent

def _jvp_pre_process_primals(primals, tangents, use_graph):
    tmp = primals
    primals = []
    for primal, tangent in zip(tmp, tangents):
        if not use_graph:
            primal = primal.clone().detach()
        primal._set_grad_state(use_grad=use_graph, grad_fn=None, is_leaf=True)
        tangent._set_grad_state(use_grad=False, grad_fn=None, is_leaf=True)
        primal._set_dual_state(tangent, in_dual=True)
        primals.append(primal)
    return primals

def _jvp_args_check(primals, tangents, func, use_graph):
    assert deepnet.is_all_tensor(*primals)
    assert deepnet.is_all_tensor(*tangents)
    assert isinstance(func, FunctionType)
    assert deepnet.is_py_bool(use_graph)

def grad(inputs, outputs):
    # will take the outputs and find the
    # gradients for every input passed
    # will not accumulate them
    pass

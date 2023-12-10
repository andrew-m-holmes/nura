import deepnet


def jacobian(input, func):
    # literally computes the full jacobian matrix for func
    # which holds the partial derivatives of the func outputs wrt
    # to every single input to that func
    pass


def vjp(primals, cotangent, func, use_graph=False):
    primals, cotangent = _vjp_pre_process(
        primals, cotangent, use_graph)
    with deepnet.use_grad():
        output = func(*primals)

    cotangents = []
    stack = [(output.grad_fn, cotangent)]
    while stack:
        node, cotangent = stack.pop()
        if _is_leaf_node(node):
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
            use_grad=False, grad_fn=None, is_leaf=False)
    return output, tuple(reversed(cotangents))


def _vjp_pre_process(primals, cotangent, use_graph):
    # TODO assert diff
    assert all(deepnet.is_tensor(primal) for primal in primals)
    assert deepnet.is_tensor(cotangent)

    temp = primals
    primals = []
    for primal in temp:
        if not use_graph:
            primal = primal.detach().clone()
        primal._set_grad_state(
            use_grad=True, grad_fn=None, is_leaf=True)
        primals.append(primal)
    cotangent._set_grad_state(
        use_grad=True, grad_fn=None, is_leaf=True)
    return primals, cotangent


def _is_leaf_node(node):
    if node is not None:
        return repr(node) == "AccumulateGrad"
    return False


def _is_intermediate_node(node):
    if node is not None:
        return "Backward" in repr(node)
    return False


def jvp(primals, tangents, func, use_graph=False):
    dual_tensors = _jvp_pre_process(primals, tangents, use_graph)
    with deepnet.forward_ad():
        with deepnet.set_grad(use_graph):
            output = func(*dual_tensors)
    return _jvp_post_process(output)


def _jvp_post_process(output):
    if isinstance(output, tuple):
        return tuple(dual_tensor.unpack() for dual_tensor in output)
    return output.unpack()


def _jvp_pre_process(primals, tangents, use_graph):
    # TODO assert diff
    assert all(deepnet.is_tensor(primal) for primal in primals)
    assert all(deepnet.is_tensor(tangent) for tangent in tangents)
    assert len(primals) == len(tangents)

    dual_tensors = []
    for primal, tangent in zip(primals, tangents):
        if use_graph:
            primal._set_grad_state(
                use_grad=True, grad_fn=None, is_leaf=True)
        dual_tensor = deepnet.dual_tensor(primal, tangent)
        dual_tensors.append(dual_tensor)
    return dual_tensors


def grad(inputs, outputs):
    # will take the outputs and find the
    # gradients for every input passed
    # will not accumulate them
    pass

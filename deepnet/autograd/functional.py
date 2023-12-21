import deepnet


def jacobian(input, func):
    # literally computes the full jacobian matrix for func
    # which holds the partial derivatives of the func outputs wrt
    # to every single input to that func
    pass


def vjp(primals, cotangent, func, use_graph=False):
    _vjp_args_check(primals, cotangent, use_graph)
    primals, cotangent = _vjp_pre_process(
        primals, cotangent, use_graph)
    with deepnet.use_grad():
        output = func(*primals)
        output.backward(cotangent)
    cotangents = [tensor.grad for tensor in primals]
    output = _vjp_post_process_output(output, use_graph)
    return output, cotangents

def _vjp_post_process_output(output, use_graph):
    if not use_graph:
        output._set_grad_state(
            use_grad=False, grad_fn=None, is_leaf=True)
    return output


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


def _vjp_args_check(primals, cotangent, use_graph):
    assert deepnet.is_all_tensor(*primals)
    assert deepnet.is_tensor(cotangent)
    assert deepnet.is_py_bool(use_graph)
    assert all(tensor.dtype.differentiable() for tensor in primals)
    assert cotangent.dtype.differentiable()


def jvp(primals, tangents, func, use_graph=False):
    # TODO
    pass


def _jvp_post_process(output):
    # TODO
    pass


def _jvp_pre_process(primals, tangents, use_graph):
    # TODO
    pass


def grad(inputs, outputs):
    # will take the outputs and find the
    # gradients for every input passed
    # will not accumulate them
    pass

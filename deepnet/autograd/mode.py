from contextlib import contextmanager


def usegrad():
    return _Autograd._usegrad


def revmode():
    return _Autograd._revmode


class _Autograd:

    _usegrad = True
    _revmode = True


@contextmanager
def autograd(state=True, rev=True):
    prevstate = _Autograd._usegrad
    prevmode = _Autograd._revmode
    _Autograd._usegrad = state
    _Autograd._revmode = rev
    try:
        yield
    finally:
        _Autograd._usegrad = prevstate
        _Autograd._revmode = prevmode

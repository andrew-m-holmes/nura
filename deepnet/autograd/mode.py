from contextlib import contextmanager


def usegrad():
    return _Autograd._enabled


def forwardmode():
    return _Autograd._forward


def reversemode():
    return _Autograd._reverse


class _Autograd:

    _enabled = True
    _reverse = True
    _forward = False


@contextmanager
def autograd(enabled=True, reverse=True, forward=False):
    prev_enabled = _Autograd._enabled
    prev_reverse = _Autograd._reverse
    prev_forward = _Autograd._forward
    _Autograd._enabled = enabled
    _Autograd._forward = forward
    _Autograd._reverse = reverse
    try:
        yield
    finally:
        _Autograd._enabled = prev_enabled
        _Autograd._forward = prev_forward
        _Autograd._reverse = prev_reverse

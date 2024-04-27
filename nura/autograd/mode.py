from contextlib import contextmanager


def usegrad():
    return _Autograd._enabled


def forwardmode():
    return _Autograd._forward and _Autograd._enabled


def reversemode():
    return _Autograd._reverse and _Autograd._enabled


class _Autograd:

    _enabled = True
    _reverse = True
    _forward = False


@contextmanager
def grad():
    yield autograd(enabled=True)


@contextmanager
def nograd():
    yield autograd(enabled=False)


@contextmanager
def forwardgrad():
    yield autograd(enabled=True, reverse=False, forward=True)


@contextmanager
def reversegrad():
    yield autograd(enabled=True, reverse=True, forward=False)


@contextmanager
def autograd(enabled=True, reverse=True, forward=False):
    if reverse and forward:
        raise ValueError("Cannot be in reverse mode and forwade mode simultaneously")

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

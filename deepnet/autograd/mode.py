from contextlib import contextmanager


def usegrad():
    return Autograd._usegrad


class Autograd:

    _usegrad = True


@contextmanager
def autograd(state=True):
    prev = Autograd._usegrad
    Autograd._usegrad = state
    try:
        yield
    finally:
        Autograd._usegrad = prev

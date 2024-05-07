from contextlib import contextmanager


class Autograd:
    _usegrad = True

    @classmethod
    def enabled(cls):
        return cls._usegrad


@contextmanager
def usegrad():
    usegrad = Autograd._usegrad
    Autograd._usegrad = True

    try:
        yield
    finally:
        Autograd._usegrad = usegrad


@contextmanager
def nograd():
    usegrad = Autograd._usegrad
    Autograd._usegrad = False

    try:
        yield
    finally:
        Autograd._usegrad = usegrad


@contextmanager
def setgrad(state: bool):
    usegrad = Autograd._usegrad
    Autograd._usegrad = state

    try:
        yield
    finally:
        Autograd._usegrad = usegrad

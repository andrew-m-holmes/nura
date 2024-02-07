from contextlib import contextmanager


def usegrad():
    return Autograd._usegrad

def revmode():
    return Autograd._rev


class Autograd:

    _usegrad = True
    _rev = True


@contextmanager
def autograd(state=True, rev=True):
    prevstate = Autograd._usegrad
    prevmode = Autograd._rev
    Autograd._usegrad = state
    Autograd._rev = rev
    try:
        yield
    finally:
        Autograd._usegrad = prevstate
        Autograd._rev = prevmode

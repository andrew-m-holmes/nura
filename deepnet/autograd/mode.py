from contextlib import contextmanager

def gradon():
    return Autograd._autograd

class Autograd:

    _autograd = True


@contextmanager
def autograd(state=True):
    prev = Autograd._autograd
    Autograd._autograd= state
    try:
        yield
    finally:
        Autograd._autograd = prev




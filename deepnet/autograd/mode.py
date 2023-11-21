from contextlib import contextmanager


class Grad:

    _reverse = True
    _use_grad = True

    @classmethod
    def enabled(cls):
        return cls._use_grad


@contextmanager
def no_grad():
    prev = Grad._use_grad
    Grad._use_grad = False
    try:
        yield
    finally:
        Grad._use_grad = prev


@contextmanager
def use_grad():
    prev = Grad._use_grad
    Grad._use_grad = True
    try:
        yield
    finally:
        Grad._use_grad = prev


@contextmanager
def set_grad(mode):
    prev = Grad._use_grad
    Grad._use_grad = mode
    try:
        yield
    finally:
        Grad._use_grad = prev

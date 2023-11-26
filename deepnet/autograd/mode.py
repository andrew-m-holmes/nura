from contextlib import contextmanager


class Grad:

    _reverse = True
    _use_grad = True

    @classmethod
    def enabled(cls):
        return cls._use_grad

    @classmethod
    def in_forward_mode(cls):
        return not cls._reverse

    @classmethod
    def in_reverse_mode(cls):
        return cls._reverse


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
def set_grad(value):
    prev = Grad._use_grad
    Grad._use_grad = value
    try:
        yield
    finally:
        Grad._use_grad = prev


@contextmanager
def forward_autograd():
    prev = Grad._use_grad
    prev_mode = Grad._reverse
    Grad._use_grad = True
    Grad._reverse = False
    try:
        yield
    finally:
        Grad._use_grad = prev
        Grad._reverse = prev_mode

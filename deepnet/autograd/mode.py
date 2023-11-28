from contextlib import contextmanager


class Autograd:

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
    prev = Autograd._use_grad
    Autograd._use_grad = False
    try:
        yield
    finally:
        Autograd._use_grad = prev


@contextmanager
def use_grad():
    prev = Autograd._use_grad
    Autograd._use_grad = True
    try:
        yield
    finally:
        Autograd._use_grad = prev


@contextmanager
def set_grad(value):
    prev = Autograd._use_grad
    Autograd._use_grad = value
    try:
        yield
    finally:
        Autograd._use_grad = prev


@contextmanager
def forward_autograd():
    prev = Autograd._use_grad
    prev_mode = Autograd._reverse
    Autograd._use_grad = True
    Autograd._reverse = False
    try:
        yield
    finally:
        Autograd._use_grad = prev
        Autograd._reverse = prev_mode


@contextmanager
def reverse_autograd():
    prev = Autograd._use_grad
    prev_mode = Autograd._reverse
    Autograd._use_grad = True
    Autograd._reverse = True
    try:
        yield
    finally:
        Autograd._use_grad = prev
        Autograd._reverse = prev_mode

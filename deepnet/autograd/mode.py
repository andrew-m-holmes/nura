from contextlib import contextmanager


class Autograd:

    _use_grad = True
    _forward_mode = False

    @classmethod
    def grad_enabled(cls):
        return cls._use_grad

    @classmethod
    def forward_ad_enabled(cls):
        return cls._forward_mode


@contextmanager
def use_grad():
    prev = Autograd._use_grad
    Autograd._use_grad = True
    try:
        yield
    finally:
        Autograd._use_grad = prev


@contextmanager
def no_grad():
    prev = Autograd._use_grad
    Autograd._use_grad = False
    try:
        yield
    finally:
        Autograd._use_grad = prev


@contextmanager
def set_grad(mode):
    prev = Autograd._use_grad
    Autograd._use_grad = mode
    try:
        yield
    finally:
        Autograd._use_grad = prev


@contextmanager
def forward_ad():
    Autograd._forward_mode = True
    try:
        yield
    finally:
        Autograd._forward_mode = False

from contextlib import contextmanager


def grad_enabled():
    return Autograd._use_grad


def forward_ad_enabled():
    return Autograd._forward_mode


class Autograd:

    _use_grad = True
    _forward_mode = False


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

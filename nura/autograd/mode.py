import nura
from nura.tensors import Tensor
from contextlib import contextmanager
from typing import Generator


class Autograd:
    _usegrad = True
    _forwardmode = False

    @classmethod
    def reversemode(cls) -> bool:
        return cls._usegrad and not cls._forwardmode

    @classmethod
    def forwardmode(cls) -> bool:
        return cls._forwardmode and not cls._usegrad


@contextmanager
def usegrad() -> Generator:
    usegrad = Autograd._usegrad
    forwardmode = Autograd._forwardmode
    Autograd._usegrad = True
    Autograd._forwardmode = False
    try:
        yield
    finally:
        Autograd._usegrad = usegrad
        Autograd._forwardmode = forwardmode


@contextmanager
def nograd() -> Generator:
    usegrad = Autograd._usegrad
    Autograd._usegrad = False
    try:
        yield
    finally:
        Autograd._usegrad = usegrad


@contextmanager
def setgrad(state: bool) -> Generator:
    usegrad = Autograd._usegrad
    forwardmode = Autograd._forwardmode
    Autograd._usegrad = state
    Autograd._forwardmode = not state
    try:
        yield
    finally:
        Autograd._usegrad = usegrad
        Autograd._forwardmode = forwardmode


@contextmanager
def forwardmode() -> Generator:
    usegrad = Autograd._usegrad
    forwardmode = Autograd._forwardmode
    Autograd._usegrad = False
    Autograd._forwardmode = True
    try:
        yield
    finally:
        Autograd._usegrad = usegrad
        Autograd._forwardmode = forwardmode

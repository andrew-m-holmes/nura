import nura
from nura.tensors import Tensor
from contextlib import contextmanager
from typing import Generator


class Autograd:
    _usegrad = True
    _forwardmode = False

    @classmethod
    def enabled(cls) -> bool:
        return cls._usegrad

    @classmethod
    def disabled(cls) -> bool:
        return not cls._usegrad

    @classmethod
    def forwardmode(cls) -> bool:
        return cls._forwardmode

    @classmethod
    def reversead(cls) -> bool:
        return not cls._forwardmode


@contextmanager
def usegrad() -> Generator:
    usegrad = Autograd._usegrad
    Autograd._usegrad = True
    try:
        yield
    finally:
        Autograd._usegrad = usegrad


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
    Autograd._usegrad = state
    try:
        yield
    finally:
        Autograd._usegrad = usegrad


@contextmanager
def forwardmode() -> Generator:
    forwardmode = Autograd._forwardmode
    Autograd._forwardmode = True
    try:
        yield
    finally:
        Autograd._forwardmode = forwardmode
        nura.forwardad.cleanup()

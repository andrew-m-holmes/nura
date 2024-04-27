import numpy as np
from .types import char, byte, short, int, long, half, float, double, bool, dtypeof, inf
from .autograd.mode import (
    usegrad,
    nograd,
    autograd,
    forwardgrad,
    reversegrad,
    forwardmode,
    reversemode,
)
from .autograd.functional import grad, backward
from .tensors import tensor
from .utils import *
from .functional import *

np.set_printoptions(precision=4)

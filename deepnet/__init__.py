import numpy as np
from .tensors import tensor
from .dtype import char, byte, short, int, long, half, float, double, bool, dtypeof
from .autograd.mode import usegrad, autograd
from .autograd.functional import grad, backward
from .utils import *
from .functional import *

np.set_printoptions(precision=4)

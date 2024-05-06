import numpy as np
from .types import char, byte, short, int, long, half, float, double, bool, dtypeof, inf
from .tensors import tensor
from .autograd import backward, grad
from .utils import *
from .functional import *

np.set_printoptions(precision=4)

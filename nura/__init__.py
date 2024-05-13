import numpy as np
import nura.autograd.graph as graph
import nura.autograd.forwardad as forwardad

from .autograd.functional import backward, grad
from .autograd.mode import Autograd, usegrad, nograd, setgrad, forwardmode
from .types import char, byte, short, int, long, half, float, double, bool, dtypeof, inf
from .tensors import tensor

from .utils import *
from .functional import *


np.set_printoptions(precision=4)

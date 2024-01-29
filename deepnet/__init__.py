import numpy as np
from .tensors import Tensor, tensor
from .utils import *
from .functional import *
from .dtype import *
from .autograd.mode import gradon, autograd
from .autograd import backward

np.set_printoptions(precision=4)

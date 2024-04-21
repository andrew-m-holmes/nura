import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np


def main():

    a = np.random.rand(3)
    a_tensor = nura.tensor(a, usegrad=True)
    result_tensor = 2 - a_tensor
    result_tensor.backward(nura.oneslike(result_tensor))
    print(a_tensor.grad)


if __name__ == "__main__":
    main()

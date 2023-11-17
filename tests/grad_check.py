import deepnet
import deepnet.nn.functional as f
import numpy as np


def main():
    da = np.ones((2, 3, 5))
    db = np.ones((2, 5, 6))
    a, b = deepnet.tensor(da, use_grad=True), \
        deepnet.tensor(db, use_grad=True)
    c = a @ b
    c.backward()


if __name__ == "__main__":
    main()

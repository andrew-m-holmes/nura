import nura
import nura.nn as nn
import torch
import torch.nn as tnn


def main():

    a = nura.tensor(1.0, usegrad=True)
    b = a.bool()
    print(b.usegrad)


if __name__ == "__main__":
    main()

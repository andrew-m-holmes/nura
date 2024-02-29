import neuro
import neuro.nn as nn
import numpy as np


def main():

    class Foo(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    class Bar(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.foo = Foo()

    class Baz(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bar = Bar()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.baz = Baz()

    model = Model()
    for mod in model.allmods():
        print(mod)


if __name__ == "__main__":
    main()

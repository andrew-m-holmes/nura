import neuro
import neuro.nn as nn
import numpy as np


def main():

    class Foo(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param1 = nn.Parameter()
            self.buff1 = nn.Buffer()

    class Bar(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param2 = nn.Parameter()
            self.param3 = nn.Parameter()
            self.buff2 = nn.Buffer()
            self.foo = Foo()

    class Baz(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param4 = nn.Parameter()
            self.param5 = nn.Parameter()
            self.buff3 = nn.Buffer()
            self.buff4 = nn.Buffer()
            self.bar = Bar()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.baz = Baz()

    model = Model()
    for m in model.mods():
        print(m)

    for p in model.params():
        print(p)

    for b in model.buffs():
        print(b)

if __name__ == "__main__":
    main()

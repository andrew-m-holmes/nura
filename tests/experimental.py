import nura
import nura.nn as nn
import nura.nn.functional as f
import numpy as np

def main():
    
    x = nura.rand(5, 3, 2, 4, usegrad=True)
    p = f.softmax(x, dim=-1)
    loss = p.sum()
    loss.backward()
    print(x.grad)

if __name__ == "__main__":
    main()

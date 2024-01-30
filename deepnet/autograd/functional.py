from collections import deque

import deepnet

def grad(inpt, out, grad):
    assert out.backfn in not None
    inptmap = 





def mapify(tensors):
    if deepnet.istensor(tensors):
        tensors = (tensors,)
    return {t: deepnet.zeros_like(t) for t in tensors}


def isleaf(node):
    return isinstance(node.ctx, Leaf)

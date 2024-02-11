import deepnet

def backward(out, grad=None):
    if grad is None:
        grad = deepnet.ones_like(out)
    assert out.backfn is not None
    backfn = out.backfn
    backfn.apply(grad)
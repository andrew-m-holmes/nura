import torch
import torch.autograd.forward_ad as fwAD


class TriMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c):
        ctx.save_for_forward(a, b, c)
        return a * b * c

    @staticmethod
    def jvp(ctx, gA, gB, gC):
        a, b, c = ctx.saved_for_forward
        return gA * b * c + gB * a * c + gC * a * b


trimul = TriMul.apply
a = torch.tensor(3., requires_grad=True)
b = torch.tensor(4., requires_grad=True)
c = torch.tensor(5., requires_grad=True)

ta = torch.tensor(1.)
tb = torch.tensor(1.)
tc = torch.tensor(1.)

e = torch.tensor(5, requires_grad=True)
f = e * 2
f.backward()
with fwAD.dual_level():
    dual_a = fwAD.make_dual(a, ta)
    dual_b = fwAD.make_dual(b, tb)
    dual_c = fwAD.make_dual(c, tc)
    dual_output = trimul(dual_a, dual_b, dual_c)
    print(dual_output)

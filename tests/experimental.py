import torch
import torch.nn.functional as f


def main():

    # trying to represent a batch with 1 sample of 7 features
    x = torch.arange(20).reshape(5, 4)
    x = x.float().requires_grad_()

    # Jacobian computation
    def softmax_grad(probs):
        tensor = probs.clone().detach()
        flat = torch.flatten(tensor)
        diagonal = torch.diagflat(flat)
        off_diagonal = torch.outer(flat, flat)
        return diagonal - off_diagonal

    probs = f.softmax(x, dim=-1)
    grad = torch.ones_like(probs)
    probs.backward(grad)
    jacobian = softmax_grad(probs)
    x_grad = torch.sum(jacobian, dim=-1, keepdim=True).reshape(x.size()) * grad

    print(f"What I expected:\n{x_grad}\n")

    print(f"What autograd computed:\n{x.grad}")


if __name__ == "__main__":
    main()

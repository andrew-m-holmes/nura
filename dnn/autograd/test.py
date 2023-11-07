import torch


if __name__ == "__main__":
    a = torch.rand(3).float()
    a.requires_grad = True
    b = a * 3
    print(b)

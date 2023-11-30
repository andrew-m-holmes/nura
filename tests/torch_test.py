import torch

# Original tensor with requires_grad=True
original_tensor = torch.tensor([1.0, 2.0, 3.0])
new_tensor = original_tensor.double()
# New tensor with changed data type

print(new_tensor.requires_grad)  # False
print(new_tensor.is_leaf)        # True

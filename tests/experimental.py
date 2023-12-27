import torch
import torch.autograd.functional as taf
import deepnet.autograd.functional as daf
import numpy as np
import deepnet


def main():
    a = np.random.rand(1).astype(np.float32)
    b = np.random.rand(1).astype(np.float32)
    v = np.ones_like(a)

    a_torch_tensor = torch.from_numpy(a)
    b_torch_tensor = torch.from_numpy(b)
    v_torch_tensor = torch.from_numpy(v)

    a_deepnet_tensor = deepnet.tensor(a)
    b_deepnet_tensor = deepnet.tensor(b)
    v_deepnet_tensor = deepnet.tensor(v)

    torch_jvp_out, torch_jvp = taf.jvp(
        torch.div, (a_torch_tensor, b_torch_tensor), (v_torch_tensor, v_torch_tensor))
    print(f"Torch results:\nOutput: {torch_jvp_out}\nJVP: {torch_jvp}")
    print()
    deepnet_jvp_out, deepnet_jvp = daf.jvp(
        (a_deepnet_tensor, b_deepnet_tensor), (v_deepnet_tensor, v_deepnet_tensor), deepnet.div)
    print(f"Deepnet results:\nOutput: {deepnet_jvp_out}\nJVP: {deepnet_jvp}")

    print()
    torch_jvp_out, torch_jvp = taf.jvp(
        torch.mul, (a_torch_tensor, b_torch_tensor), (v_torch_tensor, v_torch_tensor))
    print(f"Torch results:\nOutput: {torch_jvp_out}\nJVP: {torch_jvp}")
    print()
    deepnet_jvp_out, deepnet_jvp = daf.jvp(
        (a_deepnet_tensor, b_deepnet_tensor), (v_deepnet_tensor, v_deepnet_tensor), deepnet.mul)
    print(f"Deepnet results:\nOutput: {deepnet_jvp_out}\nJVP: {deepnet_jvp}")



if __name__ == "__main__":
    main()

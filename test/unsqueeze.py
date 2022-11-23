import torch
from torch.nn import functional as F
A = torch.Tensor(
    [[1,1,1,2],
    [4,4,4,5]]
)

print(A)
print(A.unsqueeze(1))
print(A.unsqueeze(0))
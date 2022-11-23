import torch.nn as nn

w_out = nn.Linear(2, 1)
print(w_out.weight.data)
w_out.weight.data.mul_(0.1)
print(w_out.weight.data)
w_out.weight.data.mul_(0.1)
print(w_out.weight.data)
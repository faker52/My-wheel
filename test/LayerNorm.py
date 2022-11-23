import torch
from torch.nn import functional as F
A = torch.Tensor(
    [[[1,1,2],
    [2,2,1]],

    [[1,1,3],
    [2,2,10]]

     ]
)
print(A)
print(A.size())
#对所有数字做归一化

#最后的某几个维度一起归一化
ln3 = torch.nn.LayerNorm(3, elementwise_affine=False, eps=0.0)
sm = torch.nn.Softmax(dim=0) #0是一列，1、-1是一行
ln2 = sm(A)

A3 = ln3(A)
print(ln2)
print(A3)



import  torch

import torch.nn as nn
import numpy as np

a = torch.tensor([[[1.0,2.0,3.0],
                 [4.0,5.0,6.0]],
                [[1.0,2.0,3.0],
                 [4.0,5.0,6.0]]])
print(a)
print(a.shape)
ln = torch.nn.LayerNorm([2,3],elementwise_affine=False)
ln_out = ln(a)
print(ln_out)

mean = np.mean(a.numpy(), axis=(1,2))
var = np.var(a.numpy(), axis=(1,2))
div = np.sqrt(var+1e-05)
ln_out = (a.numpy()-mean[:,None,None])/div[:,None,None]
print(ln_out)

a = torch.randn((2,5))
print(a)
print(a.shape)
ln = torch.nn.LayerNorm([5],elementwise_affine=False)
ln_out = ln(a)
print(ln_out)

mean = np.mean(a.numpy(), axis=(1))
var = np.var(a.numpy(), axis=(1))
div = np.sqrt(var+1e-05)
ln_out = (a.numpy()-mean[:,None,None])/div[:,None,None]
print(ln_out)
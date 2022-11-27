import torch

a = torch.tensor([1, 2, 3.], requires_grad=True)
b = a.detach()
print(a.grad)
print(b.grad)
c = a.sum()
c.backward()
print(a.grad)
print(b.grad)
print(a)
print(b)
print(c)


'''返回：
None
tensor([0.7311, 0.8808, 0.9526], grad_fn=<SigmoidBackward>)
tensor([0.7311, 0.8808, 0.9526])
Traceback (most recent call last):
  File "test.py", line 13, in <module>
    c.sum().backward()
  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/tensor.py", line 102, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph)
  File "/anaconda3/envs/deeplearning/lib/python3.6/site-packages/torch/autograd/__init__.py", line 90, in backward
    allow_unreachable=True)  # allow_unreachable flag
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
'''
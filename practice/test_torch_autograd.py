import torch
from torch import autograd
from torch import Tensor

"""
张量计算
自动微分
模块定义和反向传播

1.torch.Tensor包含什么内容

2.torch.autograd包含什么内容？与torch.Tensor的关系什么？是怎么使用的？
2.torch.nn包含什么内容？与torch.Tensor和torch.autograd的关系什么？是怎么使用的？
"""
#1.定义张量
a = torch.ones(3, 3)
print(a)
#2.计算张量
a = torch.ones(3, 3)
b = a.cos()
print(a)
print(b)

a.cos_()
print(a)

#3.自动微分
a = torch.ones(3, 3, requires_grad = True)
b = a * a
print(b)
print(b.grad_fn)
b.backward(torch.ones_like(a))
print(a.grad)





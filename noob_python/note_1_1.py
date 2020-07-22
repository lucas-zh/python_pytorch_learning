# auto_grad
import torch

"""

"""
# 对张量进行操作
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
# .requires_grad_(...)可以改变张量的requires_grad属性,如果没有指认默认为FALSE
a=torch.randn(2,2)
a=((a*3)/(a-1))
pr
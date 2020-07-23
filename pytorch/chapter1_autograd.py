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
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a, '\n', a.requires_grad)
a.requires_grad_()
# a.requires_grad_(False)
print(a.requires_grad)
b = (a * a).sum()
print(b, b.grad_fn)
# 梯度
# 反向传播 因为out是一个纯量（scalar），out.backward()等于 out.backward(torch.tensor(1))
print('x:\n', x)
out.backward()
print('x.grad:\n', x.grad)
# vector-jacbian product
x = torch.randn(3, requires_grad=True)
print(x)
y = x * 2
i = 1
# norm : the l2 norm of the tensor, squares evrey element in tensor y, then sum them, then take a aquare root
while y.data.norm() < 1000:
    y = y * 2
    i += 1
print(i, y)

gradients=torch.tensor([0.1,1.0,0.0001],dtype=torch.float)
y.backward(gradients)
print(x.grad)
print(x.requires_grad)
print((x**2).requires_grad)
with torch.no_grad():
    print((x**2).requires_grad)


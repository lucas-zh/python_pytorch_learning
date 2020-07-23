
import torch
import numpy as np

# 创建一个 5x3 矩阵, 但是未初始化:
x = torch.empty(5, 3)
print(x)
# 创建一个随机初始化的矩阵:
x = torch.rand(5, 3)
print(x)
# 创建一个0填充的矩阵，数据类型为long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 创建tensor并使用现有数据初始化:
x = torch.tensor([5.5, 3])
print(x)
# 根据现有的张量创建张量。 这些方法将重用输入张量的属性，例如， dtype，除非设置新的值进行覆盖
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
# 获取size
print(x.size())
# note:torch.size返回值是tuple类型
# 加法1
y = torch.rand(5, 3)
print(x + y)
# 加法2
print(torch.add(x, y))
# 提供输出参数作为tensor
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# 替换
# (add x to y)
y.add_(x)
print(y)
# 可以使用与NUMPY索引方式相同的操作来对进行对张量相同的操作
print(x[:, 1])
# torch.view与numpy中reshape类似
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
# 若只有1个元素的张量,可以使用.item()来得到python数据类型的值
x = torch.rand(1)
print(x)
print(x.item())
# numpy转换
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

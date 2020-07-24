from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# print(torch.version.cuda)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10
# numpy实现两层的网络
'''

# creat random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
print(x, y)
# randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    # 更新权值
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    '''

# torch实现两层的网络
'''# creat random input and output data
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
print(x, y)
# randomly initialize weights
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    # 更新权值
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2'''
# 简单的torch实现两层的网络
'''# creat random input and output data
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# randomly initialize weights
# w1,w2是模型参数需要grad,x,y,是训练数据不需要
w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()
    # 更新权值
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
'''

# 简单的torch实现两层的网络
'''# creat random input and output data
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
)

loss_fu = nn.MSELoss(reduction='sum')
learning_rate = 1e-6
for t in range(50000):
    y_pred = model(x)
    loss = loss_fu(y_pred, y)
    print(t, loss.item())
    model.zero_grad()

    loss.backward()
    # 更新权值
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad'''

# 简单的torch实现两层的网络
'''# creat random input and output data
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
)

loss_fu = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for t in range(500):
    y_pred = model(x)
    loss = loss_fu(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()

    loss.backward()
    # 更新权值
    optimizer.step()'''

# 简单的torch实现两层的网络
# creat random input and output data
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H, )
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred


model = TwoLayerNet(D_in, H, D_out)

loss_fu = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    y_pred = model(x)
    loss = loss_fu(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()

    loss.backward()
    # 更新权值
    optimizer.step()

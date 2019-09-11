#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python 3.6
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'Lou Zehua'
__time__ = '2019/7/15 11:24'

"""
    test Funcition
"""


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 个输入通道, 6 个输出通道, 5x5 的卷积核
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 仿射运算: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 对于方形仅需要给定一个参数
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# 查看网络
print(net)

# 查看模型需要学习的参数
params = list(net.parameters())
print(len(params))
for param in params:
    print(param.size())

# 输入数据
input = Variable(torch.randn(1,1,32,32))
print(input)
out = net(input)
print(out)

# 损失函数
target = Variable(torch.arange(1, 11, dtype=torch.float32))
print(target)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

f.grad


from torch import optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


# Initialize optimizer
optimizer=torch.optim.SGD(net.parameters(),lr=1e-4,momentum=0.9)

print("Model's state_dict:")
# Print model's state_dict
for param_tensor in net.state_dict():
    print(param_tensor,"\t",net.state_dict()[param_tensor].size())
print("optimizer's state_dict:")
# Print optimizer's state_dict
for var_name in optimizer.state_dict():
    print(var_name,"\t",optimizer.state_dict()[var_name])
PATH = './test.model'
torch.save(net, PATH)
model = torch.load(PATH)
model.eval()
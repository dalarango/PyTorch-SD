import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gap_factor = nn.Linear(1, 1, bias=False)
        self.gap_2_flow = nn.Linear(1, 1, bias=False)  # implicitly including AT

    def forward(self, x):
        gap0 = self.gap_factor(torch.FloatTensor([10])) - x
        flow0 = self.gap_2_flow(gap0)
        return flow0


net = Net()
print(net)

print(list(net.parameters()))

# input = Variable(torch.randn(1, 1, 1), requires_grad=True)
# print('\nTest input:\n', input)
# out = net(input)
# print('\nTest output:\n', out)

# Output:
# Net(
#   (fc1): Linear(in_features=1, out_features=1, bias=True)
# )
# [Parameter containing:
# tensor([[0.4217]], requires_grad=True),
#  Parameter containing:
# tensor([-0.0766], requires_grad=True)]
#
# tensor([[[1.0830]]], requires_grad=True)
# tensor([[[0.3801]]], grad_fn=<AddBackward0>)
#
# 1.0830 * 0.4217 - 0.0766 = 0.3801
#
# Therefore, we know:
# the 1.0830 is the input
# the 0.4217 is the weight (randomly assigned)
# the -0.0766 is the bias  (randomly assigned)


def criterion(out, label):  # define a loss function: least squares.
    return (label - out) ** 2


# define an optimizer using Stochastic Gradient Descent
optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.5)

# define a dataset by stock behavior
behavior = pd.read_excel('behavior.xlsx')
n = 70
stock_behavior = list()
skip = 1
for i in range(n):
    if i % skip == 0:
        stock_behavior.append(behavior.iloc[i, 5])

flow_behavior = list()
for i in range(n):
    flow_behavior.append(behavior.iloc[i, 2])

dataset0 = list()
for i in range(len(stock_behavior)):
    dataset0.append((stock_behavior[i], flow_behavior[i]))
print(dataset0)

# the training loop:
print('Training...\n')
for epoch in range(1000):
    for i, data2 in enumerate(dataset0):
        X, Y = iter(data2)
        X, Y = Variable(torch.FloatTensor([X]), requires_grad=True), \
               Variable(torch.FloatTensor([Y]), requires_grad=False)
        optimizer.zero_grad()
        outputs = net(X)  # the computation graph gets executed.
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if (i == 0):
            print("Epoch {} - loss: {}".format(epoch, loss.data[0]))

# check the parameters
print(list(net.parameters()))

# making a prediction
test_in = list()
test_out = list()
test_out_stock = [100]
dt = 0.25*skip
for i in range(len(dataset0)):
    a = dataset0[i][0]
    test_in.append(a)
    b = net(Variable(torch.Tensor([test_in[i]]))).item()
    test_out.append(b)
    c = test_out_stock[i] + test_out[i]*dt
    test_out_stock.append(c)


print('\nTest input :\n', test_in)
print('\nTest output:\n', test_out)
print('\nReal output:\n', flow_behavior[:n])

plt.xlabel('Time')
plt.ylabel('Stock Behavior')
plt.axis([0, n/skip, 0, 100])
#plt.scatter(test_in, test_out, s=30, marker='o')
#plt.scatter(test_in, flow_behavior[:n], s=30, marker='x')
plt.plot(stock_behavior[:n], pickradius=10, marker='o', label='Ground Truth')
plt.plot(test_out_stock[:n], marker='x', label='Test behavior')
plt.legend()
plt.show()

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 2)  # the linear full connection function
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x

net = Net()
print(net)

print(list(net.parameters()))

input = Variable(torch.randn(1, 1, 1), requires_grad=True)
print(input)
out = net(input)
print(out)

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
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

# define a simple dataset:
data = [(1,3), (2,6), (3,9), (4,12), (5,15), (6,18)]

# the training loop:
for epoch in range(100):
    for i, data2 in enumerate(data):
        print(i)
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
test_in = 1
test_out = net(Variable(torch.Tensor([[[test_in]]])))
print(test_out)


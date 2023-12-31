import torch
import torch.nn as nn


class BaselineMNISTNetwork(nn.Module):
    """Baseline network for MNIST dataset.

    This network is the implement of baseline network for MNIST dataset, from paper
    `BadNets: Evaluating Backdooring Attackson Deep Neural Networks <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8685687&tag=1>`_.
    """

    def __init__(self):
        super(BaselineMNISTNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 10)

        self.avg_pool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        #忽略bitch-szie的情况下：
        # x:28*28*1
        # conv1: 5*5*1*16,
        x = self.conv1(x) #
        # x: 24*24*16
        x = self.relu(x)
        # x: 24*24*16
        # pooling:2*2*16
        x = self.avg_pool(x)
        # x: 12*12*16
        # conv2: 5*5*16*32
        x = self.conv2(x)
        # x: 8*8*32
        x = self.relu(x)
        # x: 8*8*32
        # pooling:2*2*32
        x = self.avg_pool(x)
        # x: 4*4*32
        x = x.contiguous().view(-1, 512)
        # x: 512*1
        # FC:512*512
        x = self.fc1(x)
        x = self.relu(x)
        # x: 512*1
        # FC:512*10    
        x = self.fc2(x)
        # x: 10*1
        # x = self.softmax(x)
        return x

if __name__ == '__main__':
    baseline_MNIST_network = BaselineMNISTNetwork()
    x = torch.randn(16, 1, 28, 28)
    x = baseline_MNIST_network(x)
    #考虑到bitch-size,则有x: 16*10*1
    print(x.size())
    print(x)

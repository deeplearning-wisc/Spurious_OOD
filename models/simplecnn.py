import torch
import torch.nn.functional as F
import torch.nn as nn


class SimpleCNN(torch.nn.Module):
    def __init__(self, hidden_dim = 20, num_classes = 2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3) # in-channel, out-channel, kernel-size
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.predict = torch.nn.Linear(hidden_dim, num_classes)   #output layer
        self.repr_dim = hidden_dim
        self.output_dim = num_classes

    def forward(self, x):
        x = F.relu(self.hidden(x))
        #x = self.bn(x)
        x = F.relu(self.hidden1(x))
        #x = self.bn(x)
        x = self.predict(x)
        # x = (F.tanh(x) * 1/2 + 1/2)*(self.ub - self.lb) + self.lb
        return x

    def get_representation(self, x):
        """
        Given input, returns representation
        "z" vector.
        """
        with torch.no_grad():
            x = x.clone()
            x = F.relu(self.hidden(x))
            x = F.relu(self.hidden1(x))
            return x
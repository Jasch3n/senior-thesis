import torch
from torch import nn
import numpy as np

from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

class OneLayerNN(nn.Module):
    """
        Basic neural net, mathematically equiavalent to a linear combination of 
        different sigmoid functions
    """
    def __init__(self, hidden_size=32, activation=nn.Sigmoid()):
        """
        Initializes model layers.
        :param input_features: The number of features of each sample.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.income = torch.nn.Linear(1, self.hidden_size)
        self.activation = activation
        self.out = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, X):
        """
        Applies the layers defined in __init__() to input features X.
        :param X: 2D torch tensor of shape [n, 11], where n is batch size.
            Represents features of a batch of data.
        :return: 2D torch tensor of shape [n, 1], where n is batch size.
            Represents prediction of wine quality.
        """
        return self.out(self.activation(self.income(X)))

class MultiLayerNN(nn.Module):
    def __init__(self, sizes, input_features=1):
        super().__init__()
        self.incoming = nn.Linear(input_features, sizes[0])
        self.hidden_layers = [self.incoming]
        self.sigmoid = nn.Sigmoid()
        for i in range(1,len(sizes)):
            self.hidden_layers.append(nn.Linear(sizes[i-1], sizes[i]))
        self.hidden_layers.append(nn.Linear(sizes[i],1))
        self.hidden = nn.Sequential(*self.hidden_layers)

    def forward(self,x):
        x = self.sigmoid(self.incoming(x))
        for layer in self.hidden_layers[1:]:
            x = self.sigmoid(layer(x))
        
        return x
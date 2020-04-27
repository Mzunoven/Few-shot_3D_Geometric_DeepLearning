import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
import torch_geometric.nn as GCN

class CNN(nn.Module):
    def __init__(self, batch_dim):
        super(CNN, self).__init__()

        # Layer 1 - conv3d - Input: 32x32x32, Output:30x30x30
        # Layer 2 - conv3d - Input: 30x30x30, Output:15x15x15
        # Layer 3 - conv3d - Input: 15x15x15, Output:13x13x13
        # Layer 4 - conv3d - Input: 13x13x13, Output:7x7x7
        # Layer 5 - Linear - Input: 7x7x7, Output:343
        self.batch_dim = batch_dim
        latent_dim = 100

        #n_out = (n_in + 2*padding - filter)/stride + 1

        # Main throughput is be a 5d tensor - (batchSize, numChannels, Depth, Height, Width)
        self.ConvModel = nn.Sequential(
            nn.Conv3d(1, 8, (3,3,3), stride=(1,1,1),padding=0), #inchannel, outchannel, kernelSize, stride, padding
            nn.ELU(),
            nn.BatchNorm3d(8), #Expects a 5d input (batchSize, numChannels, Depth, Height, Width). Arg is numChannels.
            nn.Conv3d(8, 16, (3,3,3), stride=(2,2,2),padding=1), #Layer 2
            nn.ELU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, (3,3,3), stride=(1,1,1),padding=0), #Layer 3
            nn.ELU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, (3,3,3), stride=(2,2,2),padding=1), #Layer 4
            nn.ELU(),
            nn.BatchNorm3d(64)
            #nn.Linear((7,7,7), 343), #Layer 5 - Input: (batchSize, channels=64, 7,7,7), Output: (batchSize, channels=64, 343)
            #nn.Linear(6, 10), #Layer 5 - Input: (batchSize, channels=64, 7,7,7), Output: (batchSize, channels=64, 343)
            #nn.ELU(),
            #nn.BatchNorm1d(10) #Expects a 3d input (batchSize, numChannels=64, 343)
        )

        #self.conv1 = nn.Conv3d(1, 1, 6, stride=3,padding=0)
        self.act1 = nn.LeakyReLU(0.1)


    def forward(self, x):

        #self.fc1 = nn.Linear(729*x.shape[0], x.shape[0])

        #out1 = self.conv1(x)
        #print(out1.shape)
        #out2 = torch.flatten(out1, 0, -1)
        #print(out2.shape)
        #out3 = self.fc1(out2)
        #out4 = self.act1(out3)

        self.fc1 = nn.Linear(64*6*6*6*x.shape[0], x.shape[0])

        out1 = self.ConvModel(x)
        #print(out1.shape)
        out2 = torch.flatten(out1, 0, -1)
        out3 = self.fc1(out2)
        out4 = self.act1(out3)

        return out3

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()

        self.layers = []
        self.inputs = None
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None



    def forward(self, x):
        return 0

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#from rdkit import Chem
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
#from torch_geometric.data import Data
#from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader
#import torch_geometric.nn as GCN
from torch.optim import Adam

from models import CNN
from models import GNN
from voxDataset import trainData
from voxDataset import testData

def getData(batch_dim):

    #ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.
    dataset = np.load('modelnet10.npz')
    #print(dataset.files)

    xTest = dataset['X_test'] #ndarray of size (908, 30, 30, 30)
    xTrain = dataset['X_train'] #ndarray of size (3991, 30, 30, 30)
    yTest = dataset['y_test'] #ndarray of size (908, )
    yTrain = dataset['y_train'] #ndarray of size (3991, )

    #y labels are ints ranging from 0 to 9, indicating classification as one of the models in the modelnet10 framework.
    #x arrays are 30x30x30 binary grids of either 0s or 1s, indicating the presence of a voxel in that given space.
    #Each voxel could be contained in a 24x24x24 cube. The additional space serves as padding.

    xTrain_loader = DataLoader(xTrain, batch_size=batch_dim, shuffle=True)
    yTrain_loader = DataLoader(yTrain, batch_size=batch_dim, shuffle=True)

    return xTrain_loader, yTrain_loader

def main():
    #Set device
    device = 'cpu'

    #Set hyperparameters
    batch_dim = 100
    lr = 0.001
    GNN_epochs = 100
    CNN_epochs = 10
    latent_dim = 64

    #Load data
    #x_loader = getData(batch_dim)
    #y_loader = getData(batch_dim)
    trainDataSet = trainData()
    trainLoader = DataLoader(dataset=trainDataSet, batch_size=batch_dim, shuffle=True, num_workers=0)
    testDataSet = testData()
    testLoader = DataLoader(dataset=testDataSet, batch_size=batch_dim, shuffle=True, num_workers=0)

    #We must first use a pre-trained CNN to obtain data to be used for the GCN and semantic embeddings.
    #The paper we are referencing supports the use of 'res50' or 'inception' networks. However, these are for 2d images.
    #We will instead be using the voxnet network to classify our voxel models, obtained from: https://github.com/dimatura/voxnet.
    #   **pytorch version supposedly provided here https://github.com/lxxue/voxnet-pytorch
    #Then use a GCN (input: word embeddings for every object class, output: visual classifier for every object class)

    #No easily usable pre-trained voxel CNN. We'll have to make our own.

    #CNN---------------------------------------------------------------------------------------------------------------------

    #Build Model
    voxCNN = CNN(batch_dim).to(device)

    #Define Loss Function
    loss_func = nn.MSELoss()
    #loss_func = nn.BCELoss()

    #Optimizer
    optim = Adam(voxCNN.parameters(), lr=lr)

    #for name, param in voxCNN.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data.dtype)

    #Train CNN

    epochVals = []
    lossVals = []

    for epoch in range(CNN_epochs):
        for num_batch, (x_batch, labels) in enumerate(trainLoader):

            #x_batch is of dim batch_size, x, y, z
            #labels is of dim batch_size
            x_batch = torch.unsqueeze(x_batch, 1) #unsqueeze to make it batch_size, numchannels, x, y, z

            x_batch, labels = x_batch.to(device), labels.to(device)

            x_batch = x_batch.float()
            labels = labels.float()

            #print(x_batch.shape)
            #print(labels.shape)

            CNN_pred = voxCNN(x_batch) #should return a (batch_size, ) tensor to compare w/ labels

            #print(CNN_pred.shape)
            #print(labels.shape)

            loss = loss_func(CNN_pred, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            print(num_batch)
            if (num_batch + 1) % 20 == 0:
                #print(CNN_pred)
                print("Epoch: [{}/{}], Batch: {}, Loss: {}".format(epoch, CNN_epochs, num_batch, loss.item()))
        epochVals = epochVals + [epoch]
        lossVals = lossVals + [loss]

    plt.figure()
    for i in range(len(epochVals)):
        plt.plot(epochVals, lossVals)
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.show()

    #Test the Model

    test_error = 0
    voxCNN.eval()

    x_pred = torch.empty(batch_dim)
    y_labels = torch.empty(batch_dim)

    with torch.no_grad():
        for n_batch, (x_batch, labels) in enumerate(testLoader):
            x_batch, labels = x_batch.to(device), labels.to(device)
            x_batch = torch.unsqueeze(x_batch, 1)

            x_batch = x_batch.float()
            labels = labels.float()

            pred = voxCNN(x_batch)

            x_pred = torch.cat((x_pred, pred),0)
            y_labels = torch.cat((y_labels, labels),0)

            print(n_batch)

            test_error += loss_func(pred, labels).item()

    print(test_error)

    #Plot Predictions vs. Labels


    x_pred = torch.unsqueeze(x_pred, 1)
    y_labels = torch.unsqueeze(y_labels, 1)

    xydata = torch.cat((x_pred, y_labels), 1)

    sortedlabels, indices = torch.sort(y_labels, 0)

    indices = torch.squeeze(indices)

    sortedX = torch.index_select(x_pred, 0, indices)

    #sortedxy, indices = torch.sort(xydata, 0) #Sorts by label

    #xy = sortedxy.detach().numpy()

    #print(sortedX.shape)

    #Labels near the beginning and end are often wildly off
    #Should only be of size 908, due to batch_size interference somewhere the size becomes 928

    plt.figure()
    plt.plot(range(sortedX.shape[0]), sortedX[:,0], 'r.') #pred
    plt.plot(range(sortedlabels.shape[0]), sortedlabels[:], 'b.') #labels
    plt.xlim(-1,950)
    plt.ylim(-2,12)
    plt.xlabel('Models')
    plt.ylabel('Object Class')
    plt.show()

    #Obtain Semantic Embeddings----------------------------------------------------------------------------------------------


    #GCN---------------------------------------------------------------------------------------------------------------------

    return 0

if __name__ == "__main__":
	main()

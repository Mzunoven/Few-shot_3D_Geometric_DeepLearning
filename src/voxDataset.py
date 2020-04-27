import os
import torch
import numpy as np
from torch.utils.data import Dataset

class trainData(Dataset):
    def __init__(self):
        super(trainData, self).__init__()
        #ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.

        voxData = np.load('modelnet10.npz')
        #print(dataset.files)

        xTest = voxData['X_test'] #ndarray of size (908, 30, 30, 30)
        xTrain = voxData['X_train'] #ndarray of size (3991, 30, 30, 30)
        yTest = voxData['y_test'] #ndarray of size (908, )
        yTrain = voxData['y_train'] #ndarray of size (3991, )

        #y labels are ints ranging from 0 to 9, indicating classification as one of the models in the modelnet10 framework.
        #x arrays are 30x30x30 binary grids of either 0s or 1s, indicating the presence of a voxel in that given space.
        #Each voxel could be contained in a 24x24x24 cube. The additional space serves as padding.

        self.xData = xTrain
        self.yData = yTrain


    def __getitem__(self, idx):
        return self.xData[idx], self.yData[idx]

    def __len__(self):
        return len(self.yData)

if __name__ == "__main__":
    dataset = trainData()

class testData(Dataset):
    def __init__(self):
        super(testData, self).__init__()
        #ModelNet10 provides a dataset of models in the form of .OFF files. A voxelized form of the dataset was provided by http://aguo.us/writings/classify-modelnet.html.

        voxData = np.load('modelnet10.npz')
        #print(dataset.files)

        xTest = voxData['X_test'] #ndarray of size (908, 30, 30, 30)
        xTrain = voxData['X_train'] #ndarray of size (3991, 30, 30, 30)
        yTest = voxData['y_test'] #ndarray of size (908, )
        yTrain = voxData['y_train'] #ndarray of size (3991, )

        #y labels are ints ranging from 0 to 9, indicating classification as one of the models in the modelnet10 framework.
        #x arrays are 30x30x30 binary grids of either 0s or 1s, indicating the presence of a voxel in that given space.
        #Each voxel could be contained in a 24x24x24 cube. The additional space serves as padding.

        self.xData = xTest
        self.yData = yTest

    def get_labels(self):
        return self.yData

    def __getitem__(self, idx):
        return self.xData[idx], self.yData[idx]

    def __len__(self):
        return len(self.yData)

if __name__ == "__main__":
    dataset = trainData()
        
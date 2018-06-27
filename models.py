## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.batchn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,64,5)
        self.batchn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,5)
        self.batchn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,5)
        self.batchn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256,512,5)
        self.batchn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(4608,2000)
        self.batchn6 = nn.BatchNorm1d(2000)
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(2000,1000)
        self.batchn7 = nn.BatchNorm1d(1000)
        self.drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(1000,400)
        self.batchn8 = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(400,200) 
        self.batchn9 = nn.BatchNorm1d(200)
        self.fc5 = nn.Linear(200,136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.batchn1(self.conv1(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.batchn2(self.conv2(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.batchn3(self.conv3(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.batchn4(self.conv4(x))))
        x = self.drop(x)
        x = self.pool(F.relu(self.batchn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = F.relu(self.batchn6(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.batchn7(self.fc2(x)))
        x = self.drop(x)
        x = F.relu(self.batchn8(self.fc3(x)))
        x = self.drop(x)
        x = F.relu(self.batchn9(self.fc4(x)))
        x = self.drop(x)
        x = self.fc5(x)
        # a softmax layer to convert the 10 outputs into a distribution of class scores
           
        # a modified x, having gone through all the layers of your model, should be returned
        return x

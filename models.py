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
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,3)
        self.conv4 = nn.Conv2d(128,256,3)
        self.linear1 = nn.Linear(11*11*256,272)
        #self.linear2 = nn.Linear(512,512)
        self.linear3 = nn.Linear(272,136)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2,2)
        self.drop2 = nn.Dropout2d(0.3)
        self.drop1 = nn.Dropout(0.3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        m = nn.LeakyReLU(0.05)
        l = nn.Tanh()
        x = self.pool(m(self.batch_norm1(self.conv1(x))))
        x = self.drop2(x)
        x = self.pool(m(self.batch_norm2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool(m(self.batch_norm3(self.conv3(x))))
        x = self.drop2(x)
        x = self.pool(m(self.batch_norm4(self.conv4(x))))
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.drop1(x)
        #x = l(self.linear2(x))
        #x = self.drop1(x)
        x = self.linear3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

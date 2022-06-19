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
        self.conv1 = nn.Conv2d(1, 32, 5) #That's our first conv layer, 224*224*1 (Input) and 220*220*32 (output), 
        #due to new_n = ((n+2p-f)/s) + 1
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2, 2) #110*110*32 (output)
        
        self.conv2 = nn.Conv2d(32, 64, 5) #That's our second conv layer, 110*110*32 (Input) and 106*106*64 (output)
        self.pool2 = nn.MaxPool2d(2, 2) #53*53*64 (output)
        
        self.conv3 = nn.Conv2d(64, 128, 4) #That's our third conv layer, 53*53*64 (Input) and 50*50*128 (output)
        self.pool3 = nn.MaxPool2d(2, 2) #25*25*128 (output)
        
        self.f_connected1 = nn.Linear(25*25*128, 512) #Fully connected layer with 512 neuron
        
        self.drop_f_connected1 = nn.Dropout(p=0.3) #Dropout with p=0.2
        
        self.f_connected2_Output = nn.Linear(512, 136) #Output layer with 136 neuron,  2 for each of the 68 keypoint (x, y) pairs
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.f_connected1(x))
        x = self.drop_f_connected1(x)
        x = self.f_connected2_Output(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

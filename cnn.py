import torch
import torch.nn as nn
import torch.nn.functional as nnfunctional


class ConvNet(torch.nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self):
        super(ConvNet, self).__init__()

        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 48, kernel_size=3, stride=1, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.batch2 = torch.nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2], padding=0)
        self.dropout1 = torch.nn.Dropout(p=0.25, inplace=False)
        self.conv3 = torch.nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.batch3 = torch.nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=[1, 2], stride=[1, 2], padding=0)
        self.dropout2 = torch.nn.Dropout(p=0.25, inplace=False)
        # 4608 input features, 64 output features (see sizing flow below)
        #self.fc1 = torch.nn.Linear(46*96, 768)
        self.fc1 = torch.nn.Linear(22 * 96, 512)
        self.dropout3 = torch.nn.Dropout(p=0.5, inplace=False)
        # 64 input features, 10 output features for our 10 defined classes
        #self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (1, 5, 185) to (48, 5, 185)
        #print("input: " + str(x.shape))
        x = nnfunctional.relu(self.conv1(x))
        x = self.batch1(x)
        #print(x.shape)
        x = nnfunctional.relu(self.conv2(x))
        x = self.batch2(x)
        #print(x.shape)
        # Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool1(x)
        x = self.dropout1(x)
        #print(x.shape)
        x = nnfunctional.relu(self.conv3(x))
        x = self.batch3(x)
        #print(x.shape)
        x = self.pool2(x)
        x = self.dropout2(x)
        #print(x.shape)
        # Reshape data to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        dims = x.shape
        #x = x.view(-1, 18 * 16 * 16)
        x = x.view(dims[0], dims[2], dims[1]*dims[3])
        #print(x.shape)
        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        x = nnfunctional.relu(self.fc1(x))
        x = self.dropout3(x)
        #print(x.shape)
        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        #x = self.fc2(x)
        return (x)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiskMaculaNet(nn.Module):
    def __init__(self):
        super(DiskMaculaNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # New convolutional layer
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  # New convolutional layer
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 4 * 4, 512)  # Assuming input image size is 256x256
        self.fc2 = nn.Linear(512, 2)  
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))  # New pooling layer
        x = self.pool(F.relu(self.conv6(x)))  # New pooling layer
        
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
            
        return x

# Usage example:
# model = DiskMaculaNet()
# predictions = model(input_tensor)  # Input tensor shape: [batch_size, 3, 256, 256]
# The output will be: [batch_size, 4] (x_disk, y_disk, x_macula, y_macula)

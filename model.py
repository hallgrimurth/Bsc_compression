import torch
import torch.nn as nn
import torch.nn.functional as F
import detectors
import timm

def get_resnet18():
    return timm.create_model("resnet18_cifar10", pretrained=True)



class Net(nn.Module):
        def __init__(self): 
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding='same')
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')
            self.bn2 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.dropout1 = nn.Dropout(0.25)

            self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
            self.bn4 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
            self.dropout2 = nn.Dropout(0.25)

            self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same')
            self.bn6 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
            self.dropout3 = nn.Dropout(0.25)

            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(128 * 4 * 4, 128)
            self.dropout4 = nn.Dropout(0.25)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.dropout1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))))
            x = self.pool1(x)

            x = self.dropout2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(x)))))))
            x = self.pool2(x)

            x = self.dropout3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(x)))))))
            x = self.pool3(x)

            x = self.flatten(x)
            x = F.relu(self.dropout4(self.fc1(x)))
            x = self.fc2(x)

            return x
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class TutorialNet(nn.Module):
    def __init__(self):
        super(TutorialNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.relu6 = nn.ReLU()
        self.flatten = Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.relu7 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

        # self.fc1 = nn.Linear(7 * 7 * 64, 100)
        # self.relu5 = nn.ReLU()
        # self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.flatten(x)
        x = self.relu7(self.fc1(x))
        x = self.fc2(x)
        return x

        # x = self.relu5(self.fc1(x))
        # x = self.fc2(x)
        # return x

    
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # Adjust input size based on the image dimensions
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)  # Adjust output size for Fashion MNIST
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # dynamically calculate the size based on the size of the tensor
        x_size = x.size()[1:]
        flattened_size = 1
        for s in x_size:
            flattened_size *= s

        # flatten image input
        x = x.view(-1, flattened_size)

        # add dropout layer
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x





    

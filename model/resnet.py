import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
    """
    def __init__(self, type):
        super(binary_model, self).__init__()
        if type == 'resnet18': # size : 256
            self.base_model = models.resnet18(weights = models.ResNet18_Weights)
            self.base_model.fc = nn.Linear(512, 1)
        elif type == 'resnet34': # 256
            self.base_model = models.resnet34(weights = models.ResNet34_Weights)
            self.base_model.fc = nn.Linear(512, 1)
        elif type == 'resnet50': # 232
            self.base_model = models.resnet50(weights = models.ResNet50_Weights)
        elif type == 'resnet101': # 224
            self.base_model = models.resnet101(weights = models.ResNet101_Weights)
        elif type == 'resnet152': # 224
            self.base_model = models.resnet152(weights = models.ResNet152_Weights)
            self.base_model.fc = nn.Linear(2048, 1)
        
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    def __init__(self, type):
        super(multi_model, self).__init__()
        if type == 'resnet18': # size : 256
            self.base_model = models.resnet18(weights = models.ResNet18_Weights)
            self.base_model.fc = nn.Linear(512, 3)
        elif type == 'resnet34': # 256
            self.base_model = models.resnet34(weights = models.ResNet34_Weights)
        elif type == 'resnet50': # 232
            self.base_model = models.resnet50(weights = models.ResNet50_Weights)
        elif type == 'resnet101': # 224 
            self.base_model = models.resnet101(weights = models.ResNet101_Weights)
        elif type == 'resnet152': # 224 
            self.base_model = models.resnet152(weights = models.ResNet152_Weights)
            self.base_model.fc = nn.Linear(2048, 1)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.fc = nn.Linear(512, 3)
        
    def forward(self, x):
        return self.base_model(x)

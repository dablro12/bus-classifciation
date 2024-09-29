import torch.nn as nn
from torchvision import models
import torch
class binary_model(nn.Module):
    """ 
        Ref : https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
    """
    def __init__(self, type, num_classes =1 ):
        super(binary_model, self).__init__()
        if type == 'resnet18': # size : 256
            self.base_model = models.resnet18(weights = models.ResNet18_Weights)
            self.base_model.fc = nn.Linear(512, num_classes)
        elif type == 'resnet34': # 256
            self.base_model = models.resnet34(weights = models.ResNet34_Weights)
            self.base_model.fc = nn.Linear(512, num_classes)
        elif type == 'resnet50': # 232
            self.base_model = models.resnet50(weights = models.ResNet50_Weights)
        elif type == 'resnet101': # 224
            self.base_model = models.resnet101(weights = models.ResNet101_Weights)
        elif type == 'resnet152': # 224
            self.base_model = models.resnet152(weights = models.ResNet152_Weights)
            self.base_model.fc = nn.Linear(2048, num_classes)
        elif type == 'resnet18ch4':
            self.base_model = models.resnet18(weights = models.ResNet18_Weights)
            self.base_model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # 기존 3채널의 가중치를 사용하여 4채널에 맞게 조정
            with torch.no_grad():
                self.base_model.conv1.weight[:, :3, :, :] = self.base_model.conv1.weight[:, :3, :, :]  # 기존 RGB 가중치 복사
                self.base_model.conv1.weight[:, 3:, :, :].zero_()  # 새 채널은 0으로 초기화
            self.base_model.fc = nn.Linear(512, num_classes)
            
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    def __init__(self, type, num_classes = 3):
        super(multi_model, self).__init__()
        if type == 'resnet18': # size : 256
            self.base_model = models.resnet18(weights = models.ResNet18_Weights)
            self.base_model.fc = nn.Linear(512, num_classes)
        elif type == 'resnet34': # 256
            self.base_model = models.resnet34(weights = models.ResNet34_Weights)
        elif type == 'resnet50': # 232
            self.base_model = models.resnet50(weights = models.ResNet50_Weights)
        elif type == 'resnet101': # 224 
            self.base_model = models.resnet101(weights = models.ResNet101_Weights)
        elif type == 'resnet152': # 224 
            self.base_model = models.resnet152(weights = models.ResNet152_Weights)
            self.base_model.fc = nn.Linear(2048, num_classes)
        elif type == 'resnet18ch4':
            self.base_model = models.resnet18(weights = models.ResNet18_Weights)
            self.base_model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # 기존 3채널의 가중치를 사용하여 4채널에 맞게 조정
            with torch.no_grad():
                self.base_model.conv1.weight[:, :3, :, :] = self.base_model.conv1.weight[:, :3, :, :]  # 기존 RGB 가중치 복사
                self.base_model.conv1.weight[:, 3:, :, :].zero_()  # 새 채널은 0으로 초기화
            num_classes = 1 
            self.base_model.fc = nn.Linear(512, num_classes)
            # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        
    def forward(self, x):
        return self.base_model(x)

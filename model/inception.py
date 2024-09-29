
import torch.nn as nn
from torchvision import models

class binary_model(nn.Module):
    def __init__(self, type):
        super(binary_model, self).__init__()
        if type == 'default': # 299
            self.base_model = models.inception_v3(weights = models.Inception_V3_Weights.IMAGENET1K_V1)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 1)
        
    def forward(self, x):
        return self.base_model(x).view(-1)

class multi_model(nn.Module):
    def __init__(self, type):
        super(multi_model, self).__init__()
        if type == 'default': # 299
            self.base_model = models.inception_v3(weights = models.Inception_V3_Weights.IMAGENET1K_V1)
        # vgg16 마지막 분류기 부분을 바이너리 분류에 맞게 변경
        self.base_model.classifier[-1] = nn.Linear(1280, 3)
        
    def forward(self, x):
        return self.base_model(x)
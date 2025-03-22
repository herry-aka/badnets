import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWrapper(nn.Module):
    def __init__(self, ouput_num):
        super().__init__()
        #这里以Resnet18为例，也可以根据需要选择其他版本
        self.resnet = models.resnet18(pretrained=False)
        #修改最后一层全连接层以匹配输出类别数
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, ouput_num)

    def forward(self, x):
        return self.resnet(x)
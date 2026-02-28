import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2，只保留模型参数（官方推荐，占用空间小）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱（自建模型）
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

myModel = MyModel()
torch.save(myModel, "myModel_method1.pth")
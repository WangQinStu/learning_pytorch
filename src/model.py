import torch
from torch import nn


# 3. 搭建神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        output = self.model(x)
        return output

# 用于测试上面搭建的神经网络的正确性
if __name__ == '__main__':
    network = Network()
    test_input = torch.ones((64,3,32,32)) # 测试输入的NCWH和CIFAR10数据集一样
    test_output = network.model(test_input)
    print(test_output.shape) # 打印output的尺寸，和预计尺寸进行对比是否一致

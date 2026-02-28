import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,input):
        output = self.model1(input)
        return output


dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

model = Model()
loss_cross = CrossEntropyLoss()

for data in dataloader:
    imgs,targets = data
    outputs = model(imgs)
    # print(outputs) # 神经网络计算的输出
    # print(targets) # 真实的类别输出
    result_loss = loss_cross(outputs,targets) # 真实值和计算值之间的误差
    result_loss.backward()
    print(result_loss)



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
dataloader = DataLoader(dataset,batch_size=64)

model = Model()
loss_cross = CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(),lr=0.01)

for epco in range(20):  # 一共训练20轮
    running_loss = 0.0
    for data in dataloader: # 每一轮训练都遍历所有数据
        imgs,targets = data
        outputs = model(imgs)
        result_loss = loss_cross(outputs,targets) # 真实值和计算值之间的误差
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(f'经过第{epco}轮训练后的loss为： {running_loss}')



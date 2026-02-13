import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = Linear(196608,10)

    def forward (self,input):
        output = self.linear(input)
        return output

model = Model()
step = 0
for data in dataloader:
    imgs,target = data
    print('input image shape:',imgs.shape)
    # input = torch.reshape(imgs,(1,1,1,-1))
    input = torch.flatten(imgs) # 展平操作： 和上面效果一样
    output = model(input)

    print('output image shape:',output.shape)

import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from 资料包.源代码.nn_linear import output

# #*************1.输入tensor作为input**************
# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1],
#                       ],dtype=torch.float)
#
# # reshape成maxpool支持的格式：（batch_size, chanel, height, width）
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

# *************2.DataLoader加载数据*******************
dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.maxPool = nn.MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,x):
        output = self.maxPool(x)
        return output

# #*************1.输入tensor作为input**************
# model = Model()
# output = model(input)
# print(output)

# *************2.DataLoader加载数据*******************
model = Model()
writer = SummaryWriter('../tensorboard_log')
step = 0
for data in dataloader:
    imgs, targets = data
    output = model(imgs)
    print(output.shape)
    writer.add_images('nn_maxpool', imgs, step)
    writer.add_images('nn_maxpool_output',output,step)
    step += 1

writer.close()
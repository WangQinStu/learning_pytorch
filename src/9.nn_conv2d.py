import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=False)
dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)
        return x

model = Model()

step = 0
writer = SummaryWriter(log_dir='../tensorboard_log')

for data in dataloader:
    imgs, target = data
    output = model(imgs)
    writer.add_images('nn_conv2d',imgs,step)
    # tensorboard只能展示3个通道的图片
    # 所以需要将torch.size([64,6,30,30]) -> torch.size([xxx,3,30,30])
    # 不知道bach size可以直接写-1,程序会自动计算
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images('nn_conv2d_output', output, step)
    step += 1
    print(imgs.shape)
    print(output.shape)

writer.close()

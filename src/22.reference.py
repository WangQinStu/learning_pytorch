import torch
import torchvision
from PIL import Image
from torch import nn

# 1. 准备测试图片
img_path = 'dataset/test_imgs/cat.png'
img = Image.open(img_path)
print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)

# 2.加载网络模型
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

network = Network()
state_dict = torch.load('network_50_gpu.pth',weights_only=False,map_location=torch.device('cpu'))
network.load_state_dict(state_dict)
print(network)

# 3.开始推理
img = torch.reshape(img,(1,3,32,32)) # torch要求的输入尺寸为NCWH,所以需要reshape
network.eval()
with torch.no_grad():
    output = network(img)
print(output)

print('预测结果为： ')
match output.argmax(1).item():
    case 0:
        print('airplane')
    case 1:
        print('automobile')
    case 2:
        print('bird')
    case 3:
        print('cat')
    case 4:
        print('deer')
    case 5:
        print('dog')
    case 6:
        print('frog')
    case 7:
        print('horse')
    case 8:
        print('ship')
    case 9:
        print('truck')
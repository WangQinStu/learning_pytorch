# 1.加载预训练好的模型
# 2.修改模型使其适应新的数据集
import torchvision

# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False) # 不预训练
vgg16_true = torchvision.models.vgg16(pretrained=True) # 预训练

print('原始预训练好的vgg模型',vgg16_true)

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 在预训练好的vgg模型中添加一层
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print('在预训练好的vgg模型中添加一层后',vgg16_true)

print('未预训练的vgg模型',vgg16_false)
# 将未预训练的vgg模型中的classifier的第6层改为1个输出为10的线性层
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print('将未预训练的vgg模型中的classifier的第6层改为1个输出为10的线性层后',vgg16_false)
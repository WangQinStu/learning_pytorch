import torchvision
from torch.utils.tensorboard import SummaryWriter

trans_dataset = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize(100)
])

# 直接在数据集添加transform=trans_dataset即可转换整个数据集的格式
train_set = torchvision.datasets.CIFAR10(root='./dataset_CIFAR10',train=True,download=True,transform=trans_dataset)
test_set = torchvision.datasets.CIFAR10(root='./dataset_CIFAR10',train=False,download=True,transform=trans_dataset)

writer = SummaryWriter("tensorboard_log")
for i in range(10):
    img, target = train_set[i]
    writer.add_image('CIFAR10数据集',img,i)

print(test_set[0])
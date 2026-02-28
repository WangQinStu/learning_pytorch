# 完整的训练套路
import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

writer = SummaryWriter('./logs_network')

# 1. 准备数据集
train_dataset = torchvision.datasets.CIFAR10('./dataset',train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10('./dataset',train=False,transform=torchvision.transforms.ToTensor(),download=True)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

print('训练数据集的长度为: {}'.format(train_dataset_size))
print('测试数据集的长度为: {}'.format(test_dataset_size))

# 2. 使用DataLoader加载CIFAR10数据集，每64个一批
train_dataloader = DataLoader(train_dataset,batch_size=64)
test_dataloader = DataLoader(test_dataset,batch_size=64)

# 4. 创建网络模型
network = Network() # 从model文件中创建网络模型

# 5. 损失函数
loss_fn = nn.CrossEntropyLoss() # 分类问题可用交叉熵loss

# 6. 优化器
learning_rate = 1e-2 # 1*(10)^(-2) = 1*(1/100) = 0.01
optimizer = torch.optim.SGD(network.parameters(),lr=learning_rate)

# 7. 设置训练参数
total_train_times = 0 # 总训练次数
total_test_times = 0 # 总测试次数
epoch = 10 # 训练轮数

for i in range(epoch):
    print('----------开始第{}轮训练--------------'.format(i+1))
    # 8. 开始训练步骤
    network.train() # 当网络中有dropout、batchNorm层时有效
    for data in train_dataloader:
        imgs,targets = data # len(imgs)=64
        output = network(imgs)
        loss = loss_fn(output,targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_times += 1
        if total_train_times % 100 == 0: # 当训练次数为100的整数倍时才打印，避免无效信息
            print('训练次数： {}， loss: {}'.format(total_train_times,loss.item()))
            writer.add_scalar('trian_loss',loss,total_train_times)

    # 9. 开始测试步骤
    network.eval() # 当网络中有dropout、batchNorm层时有效
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 不反向传播计算梯度、调优
        for data in test_dataloader:
            imgs,targets = data
            output = network(imgs)
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss
            # 计算分类问题的正确率，参考test.py
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
    print(f'第{ i+1 }轮训练后，整体测试集的loss为：{total_test_loss}')
    print(f'第{ i+1 }轮训练后，整体测试集的正确率为：{total_accuracy/test_dataset_size}')
    writer.add_scalar('test_loss',loss,total_test_times)
    writer.add_scalar('test_accuracy',total_accuracy/test_dataset_size,total_test_times)
    total_test_times = total_train_times + 1

    # 10. 保存每一轮的模型
    torch.save(network,'network_{}.pth'.format(i+1)) # 保存方式1
    # torch.save(network.state_dict(),'network_{}.pth'.format(i+1)) # 保存方式2： 官方推荐
    print(f'第{i+1}轮模型已保存')
writer.close()

# 损失函数就是衡量input和target之间的差距
import torch
from torch import nn

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target,(1,1,1,3))

l1_loss = nn.L1Loss()
l1_loss = l1_loss(input,target)

mse_loss = nn.MSELoss()
mse_loss = mse_loss(input,target)


print('l1_loss: ',l1_loss)
print('mse_loss: ',mse_loss)

# 交叉熵loss
output = torch.tensor([0.1,0.2,0.3])
target = torch.tensor([1])
output = torch.reshape(output,(1,3))
cross_loss = nn.CrossEntropyLoss()
result = cross_loss(output,target)
print('cross_loss: ',result)
# 分类问题

import torch

# 预测值
outputs = torch.tensor([[0.1,0.2],
                        [0.05,0.4]])
# argmax：返回每行/列最大元素的下标
print('横向看',outputs.argmax(1))
print('竖向看',outputs.argmax(0))
preds = outputs.argmax(1)

# 实际值
targets = torch.tensor([0,1])

print('预测值=实际值？',(preds == targets))
print(f'预测值和实际值有{ (preds == targets).sum() }个相符')


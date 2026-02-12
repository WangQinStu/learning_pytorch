from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('tensorboard_log')

# 绘制y=x的标量图像
for i in range(100):
    writer.add_scalar('y=x',i,i)

# 展示图片
img_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_numpy = np.array(img_PIL) # 转化为numpy形式，因为tensorboard只接受torch.Tensor, numpy.ndarray格式的图片
print('image tpye: ',type(img_numpy))
print('shape of image:', img_numpy.shape)
writer.add_image('ant image',img_numpy,1,dataformats='HWC')

writer.close()
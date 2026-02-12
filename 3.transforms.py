from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "dataset/train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
# 1. 使用transforms将PIL的图片转为tensor
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img_PIL)
print(type(img_tensor))

# 2. 用tensorboard展示tensor类型的图片
writer = SummaryWriter('tensorboard_log')
writer.add_image('img_tensor',img_tensor)
writer.close()



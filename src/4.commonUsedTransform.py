from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


img_path = "../dataset/torch.png"
img_PIL = Image.open(img_path)
writer = SummaryWriter('../tensorboard_log')

# 1. ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img_PIL)
writer.add_image('ToTensor',img_tensor)

# 2. Normalize归一化
print('原图：', img_tensor[2][2][2])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print('归一化后：', img_norm[2][2][2])
writer.add_image('Normalize',img_norm)

# 3. Resize
print('原size:', img_tensor.size())
trans_resize = transforms.Resize((512,512))
img_resized = trans_resize(img_tensor)
print('resize后：',img_resized.size())
writer.add_image('Resize',img_resized,1)

# 3. Compose
trans_compose = transforms.Compose([trans_tensor,trans_norm,trans_resize]) # 将数组中的操作合并一起执行
img_compose = trans_compose(img_PIL)
writer.add_image('Compose',img_compose)

# 4. RandomCrop随机裁剪
trans_randomCrop = transforms.RandomCrop(256)
for i in range(10):
    img_crop = trans_randomCrop(img_tensor)
    writer.add_image('RandomCrop',img_crop,i)


writer.close()
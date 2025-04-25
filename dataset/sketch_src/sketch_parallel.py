import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
norm_layer = nn.InstanceNorm2d
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from PIL import Image, ImageFile
# ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        norm_layer(in_features)
                        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    norm_layer(64),
                    nn.ReLU(inplace=True) ]
        self.model0 = nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model1 += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        self.model1 = nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features//2
        for _ in range(2):
            model3 += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        norm_layer(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        self.model3 = nn.Sequential(*model3)

        # Output layer
        model4 = [  nn.ReflectionPad2d(3),
                        nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [nn.Sigmoid()]

        self.model4 = nn.Sequential(*model4)

    def forward(self, x, cond=None):
        x = x[:,:3,:,:]
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out
# 定义数据集
class ImageDataset(Dataset):
    def __init__(self, img_paths, sketch_paths, transform=None):
        self.img_paths = img_paths
        self.sketch_paths = sketch_paths
        self.transform = transform
        self.target_size = (1024, 512)
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # img_path = self.img_paths[idx]
            
        # sketch_path = self.sketch_paths[idx]
        # image = Image.open(img_path).convert('RGB')
        # original_size = image.size
        # image = image.resize(self.target_size)
        # if self.transform:
        #     image = self.transform(image)
        # return image, (original_size[0], original_size[1]), sketch_path, img_path
        try:
            img_path = self.img_paths[idx]
            sketch_path = self.sketch_paths[idx]
            assert os.path.basename(img_path) == os.path.basename(sketch_path)
            # print(img_path, sketch_path)
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            original_size = image.size
            image = image.resize(self.target_size)
            if self.transform:
                image = self.transform(image)
            return image, (original_size[0], original_size[1]), sketch_path, img_path
        except:
            idx = idx + 1
            print("Error in loading image at index")
            return self.__getitem__(idx)

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# # 加载模型
model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load('ckpts/model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = model1.to(device)
model1 = nn.DataParallel(model1)

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load('ckpts/model_simple_lines.pth'))
model2 = model2.to(device)
model2 = nn.DataParallel(model2)

model1.eval()
model2.eval()

# # 读取数据d
# with open('/Node11_nvme/Tsinghua_Node11/jiangzj/diffusers/IAT_dataset/test_final.json', 'r') as file:
#     items = json.load(file)
# img_paths = [item['image'] for item in items]
# sketch_paths = [item['conditioning_image'] for item in items]
import sys
img_dir = sys.argv[1]
sketch_dir = sys.argv[2]
# print(sketch_dir)
img_dirs = os.listdir(img_dir)
images = []

for dir_path in img_dirs:
    if os.path.isdir(os.path.join(img_dir, dir_path)) == False:
        continue
    for img in os.listdir(os.path.join(img_dir, dir_path)):
        if img.endswith('.jpg') or img.endswith('.png'):
            images.append(os.path.join(dir_path, img))

img_paths = []
sketch_paths = []
for image in images:
    # if os.path.exists(os.path.join(sketch_dir, image))==False:
    img_paths.append(os.path.join(img_dir, image))
    sketch_paths.append(os.path.join(sketch_dir, image))
print(images)
print(len(img_paths))
if len(img_paths) == 0:
    exit()
# for item in images:
#     if os.path.exists(item['conditioning_image']):
#         # print('-----skip:', item['conditioning_image'])
#         continue
#     img_paths.append(item['image'])
#     sketch_paths.append(item['conditioning_image'])
combined = list(zip(img_paths, sketch_paths))
import random
random.shuffle(combined)
list1_shuffled, list2_shuffled = zip(*combined)
img_paths = list(list1_shuffled)
sketch_paths = list(list2_shuffled)

dataset = ImageDataset(img_paths, sketch_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
var = 'Simple Lines'
# 并行推理
for batch in tqdm(dataloader):
    images, original_sizes, sketch_paths, image_paths = batch
    images = images.to(next(model2.parameters()).device)
    with torch.no_grad():
        if var == 'Simple Lines':
            drawings = model2(images)
        
    drawings = drawings.detach().cpu()
    for i, drawing in enumerate(drawings):
        # Resize back to original size
        
        sketch_path = sketch_paths[i]
        # print(f"Processed and saved: {sketch_path} and Image path:{image_paths[i]}" )
        drawing = transforms.ToPILImage()(drawing)
        drawing = drawing.resize((original_sizes[0][i], original_sizes[1][i]))
        
        if not os.path.exists(sketch_path):
            os.makedirs(os.path.dirname(sketch_path), exist_ok=True)
        drawing.save(sketch_path)
        
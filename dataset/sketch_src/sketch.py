import numpy as np
import torch
import torch.nn as nn
# import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
norm_layer = nn.InstanceNorm2d
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

model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load('model.pth', map_location=torch.device('cuda')))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load('model_simple_lines.pth', map_location=torch.device('cuda')))
model2.eval()

def predict(input_img, ver):
    input_img = Image.open(input_img)
    transform = transforms.Compose([ transforms.ToTensor()])
    input_img = transform(input_img)
    input_img = torch.unsqueeze(input_img, 0)

    drawing = 0
    with torch.no_grad():
        # import pdb
        # pdb.set_trace()
        print(input_img.shape)
        if ver == 'Simple Lines':
            drawing = model2(input_img)[0].detach()
        else:
            drawing = model1(input_img)[0].detach()
    
    drawing = transforms.ToPILImage()(drawing)
    return drawing
    
# for i in tqdm(range(0,100)):
#     var = 'Complex Lines'
#     input_img = f"/data1/jiangzj/code/IAT/iat_data/image/{i:05d}.jpg"
#     img = predict(input_img, var)
#     img.save(f'/data1/jiangzj/code/IAT/lineart_img/{i:05d}_{var}.jpg')

# input_folder = '/Node11_nvme/jiangzj/diffusers/data/laion-art-high-resolution'
# output_folder = '/Node11_nvme/jiangzj/diffusers/data/sketches_1025_sketches'

# os.makedirs(output_folder, exist_ok=True)
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith('.jpeg') or filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
#         img_path = os.path.join(input_folder, filename)
        
#         name, ext = os.path.splitext(filename)
#         new_filename = name + ext
        
#         output_path = os.path.join(output_folder, new_filename)
#         img = predict(img_path, var)    
#         img.save(output_path)
#         print(f"Processed and saved: {output_path}")
var = 'Simple Lines'
image_extensions = ('.jpg', '.jpeg', '.png')
img_paths = []
# output_paths = []
# folder_path = '/Node11_nvme/jiangzj/diffusers/data/laion-art-high-resolution/laion-art-high-resolution/laion-art/'
import json
with open('/Node11_nvme/Tsinghua_Node11/jiangzj/diffusers/IAT_dataset/train_final.json', 'r') as file:
    items = json.load(file)
import random
random.shuffle(items)
for item in items:
    img_path = item['image']
    sketch_path = item['conditioning_image']
    if os.path.exists(sketch_path):
        continue
    try:
        os.makedirs(os.path.dirname(sketch_path), exist_ok=True)
        if os.path.exists(sketch_path):
            continue
        img = predict(img_path, var)    
        img.save(sketch_path)
        print(f"Processed and saved: {sketch_path}")
    except:
        print(f"Error processing {sketch_path}")

# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         # 检查文件扩展名是否为图片格式
#         if file.lower().endswith(image_extensions):
#             # 构造原始文件路径
#             original_path = os.path.join(root, file)
#             # 构造新的文件名（在源文件名最后加上sketch）
#             new_filename = os.path.splitext(file)[0] + '_sketch' + os.path.splitext(file)[1]
#             # 构造新的文件路径
#             img_paths.append(original_path)
            
#             new_path = os.path.join(root, new_filename)
#             output_paths.append(new_path)
            
#             print(new_path)
# assert len(img_paths) == len(output_paths)
# from tqdm import tqdm
# for img_path, output_path in tqdm(zip(img_paths, output_paths)):
#     try:
#         if os.path.exists(output_path):
#             continue
        
#         img = predict(img_path, var)    
#         img.save(output_path)
#         print(f"Processed and saved: {output_path}")
#     except:
#         print(f"Error processing {output_path}")
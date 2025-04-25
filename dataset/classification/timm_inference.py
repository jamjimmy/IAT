import pdb
import argparse
import torch
from timm_train import val_transform, device  # 根据需要导入
import timm
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
import time
# 定义安全的图像加载函数
def safe_image_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"加载图像 {path} 时出错: {e}")
        return Image.new('RGB', (224, 224))  # 返回一个空白图像

# 加载模型
def load_model(checkpoint_path):
    model_name = 'resnet50d'
    model = timm.create_model(model_name, pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 二分类
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# 推理函数
def infer(model, image_paths):
    print(f"推理批次大小: {len(image_paths)}")
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(lambda path: val_transform(safe_image_loader(path)), image_paths))
    
    batch = torch.stack(images).to(device)
    print(f"批次张量形状: {batch.shape}")

    with torch.no_grad():
        outputs = model(batch)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.cpu().numpy()

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = safe_image_loader(self.image_paths[idx])
        img = self.transform(img)
        return img, self.image_paths[idx]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='模型推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--image_dir', type=str, required=True, help='要推理的图像目录路径')
    parser.add_argument('--batch_size', type=int, default=128, help='推理时的批次大小')
    
    args = parser.parse_args()
    
    model = load_model(args.checkpoint)
    
    if not os.path.isdir(args.image_dir):
        print(f"提供的路径不是一个目录: {args.image_dir}")
        exit(1)
    
    # temp

    image_files = []
    for root, dirs, files in os.walk(args.image_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) :
                image_files.append(os.path.join(root, f))


    #temp

    if not image_files:
        print("目录中没有找到支持的图像文件。")
        exit(1)
    
    batch_size = args.batch_size
    total_images = len(image_files)
    dataset = ImageDataset([os.path.join(args.image_dir, f) for f in image_files], val_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # start_time = time.time()
    # file_count = 0
    # appearance_paths = []
    # interior_paths = []
    output_dir = '/Node09_nvme/qinyf/iat_electron/interior'
    
    for batch_imgs, batch_files in dataloader:
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            outputs = model(batch_imgs)
            _, predicted = torch.max(outputs.data, 1)
        
        for file, pred in zip(batch_files, predicted.cpu().numpy()):
            print(f'{file}: {"appearance" if pred == 0 else "interior"}')
            if pred != 0:
                if not os.path.exists(os.path.join(output_dir, os.path.basename(file))):
                    os.symlink(file, os.path.join(output_dir, os.path.basename(file)))


        # end_time = time.time()
        # inference_time = end_time - start_time
        # file_count += len(batch_files)
        # print(f'推理时间: {inference_time:.4f} 秒, 平均每张图片时间: {inference_time / file_count:.4f} 秒, 总图片数: {file_count}')


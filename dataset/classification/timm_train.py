import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
from tqdm import tqdm  # 导入 tqdm 库

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 初始化模型为 None
model = None  

# 确定使用哪张 GPU
gpu_id = 2  
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),            
    transforms.RandomHorizontalFlip(),        
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),            
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])  
])

# 加载自定义数据集
train_dataset = datasets.ImageFolder(root='/data/Tsinghua/qinyf/classification/car_class/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='/data/Tsinghua/qinyf/classification/car_class/test', transform=val_transform)

# 定义安全的图像加载函数
def safe_image_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return Image.new('RGB', (224, 224))  # 返回一个空白图像

# 替换默认的图像加载函数
datasets.folder.default_loader = safe_image_loader

train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# 加载预训练模型并修改分类头
def get_model(model_name='resnet50d', num_classes=2):
    model = timm.create_model(model_name, pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 修改为二分类
    model = model.to(device)
    return model

# 定义损失函数和优化器
def get_criterion_optimizer(model, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()  # 设为训练模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 显示进度条
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条描述
            loop.set_postfix(loss=running_loss/total, accuracy=100. * correct / total)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

        # 保存中间训练权重
        checkpoint_path = f'checkpoints/electron_model_epoch_{epoch+1}.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"已保存模型权重到 {checkpoint_path}")

# 模型评估
def evaluate_model(model, val_loader):
    model.eval()  # 设为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    model = get_model()
    criterion, optimizer = get_criterion_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)
    evaluate_model(model, val_loader)

    # 可视化部分训练数据
    dataiter = iter(train_loader)
    try:
        images, labels = next(dataiter)
        print(' '.join(f'{train_dataset.classes[labels[j]]}' for j in range(min(4, len(labels)))))
    except StopIteration:
        print("数据加载器为空，无法获取样本。")
    
    print(' '.join(f'{train_dataset.classes[labels[j]]}' for j in range(4)))
import os
from PIL import Image
import numpy as np
import clip
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import argparse

class YourDataset(Dataset):
    def __init__(self, img_root, meta_root, is_train, preprocess):
        self.img_root = img_root
        self.meta_root = meta_root
        self.train_set_file = os.path.join(meta_root, 'train.txt')
        self.test_set_file = os.path.join(meta_root, 'test.txt')
        self.is_train = is_train
        self.img_process = preprocess
        self.samples = []
        self.sam_labels = []
        self.read_file = self.train_set_file if is_train else self.test_set_file
        with open(self.read_file, 'r') as f:
            for line in f:
                img_path = os.path.join(os.path.dirname(self.img_root), line.strip())
                label = line.strip().split('/')[2]
                label = label.replace("_", " ")
                label = "photo of " + label
                self.samples.append(img_path)
                self.sam_labels.append(label)
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        token = self.tokens[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.img_process(image)
        return image, token

# 解析命令行参数
parser = argparse.ArgumentParser(description="训练模型")
parser.add_argument('--epochs', type=int, default=10, help="总的训练epoch数")
args = parser.parse_args()

# 定义起始epoch
st = 0

# 创建模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net, preprocess = clip.load("RN50", device=device, jit=False)

optimizer = optim.Adam(net.parameters(), lr=1e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 创建损失函数
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# 加载数据集
your_dataset = YourDataset(img_root='car_class', meta_root='car_class', is_train=True, preprocess=preprocess)
dataset_size_your = len(your_dataset)
your_dataloader = DataLoader(your_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=False)

phase = "train"
model_name = "car_classify"
ckt_gap = 4
for epoch in range(st, args.epochs):
    scheduler.step()
    total_loss = 0
    batch_num = 0
    with torch.cuda.amp.autocast(enabled=True):
        for images, label_tokens in your_dataloader:
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == "train"):
                logits_per_image, logits_per_text = net(images, label_tokens)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                # Convert logits to float32 for loss calculation
                logits_per_image = logits_per_image.to(torch.float32)
                logits_per_text = logits_per_text.to(torch.float32)
                # if logits_per_image.dtype == torch.float16:
                    # ground_truth = ground_truth.to(torch.float16)
                cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
                total_loss += cur_loss
                if phase == "train":
                    cur_loss.backward()
                    optimizer.step()
                    if device != "cpu":
                        clip.model.convert_weights(net)
            if batch_num % 4 == 0:
                logger.info('{} epoch:{} loss:{}'.format(phase, epoch, cur_loss))
        epoch_loss = total_loss / len(your_dataloader)
        torch.save(net.state_dict(), f"{model_name}_epoch_{epoch}.pth")
        logger.info(f"weights_{epoch} saved")
        if epoch % ckt_gap == 0:
            checkpoint_path = f"{model_name}_ckt.pth"
            checkpoint = {
                'it': epoch,
                'network': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"checkpoint_{epoch} saved")
        logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
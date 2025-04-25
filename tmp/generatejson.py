import os
import json
import sys
from PIL import Image
image_dirs = sys.argv[1]
sketch_dirs = sys.argv[2]
length = 700
all_len = 1000
items = []
try:
    for image_dir in os.listdir(image_dirs):
        if os.path.isdir(os.path.join(image_dirs, image_dir)) == False:
            continue
        with open(os.path.join(image_dirs, image_dir, 'caption.txt'), 'r') as f:
                caption = f.read()
        for image in os.listdir(os.path.join(image_dirs, image_dir)):
            if image.endswith('.jpg') or image.endswith('.png'):
                image_path = os.path.join(image_dirs, image_dir, image)
                sketch_path = os.path.join(sketch_dirs, image_dir, image)
                if os.path.exists(sketch_path):
                    data = {
                        'image': image_path,
                        'conditioning_image': sketch_path,
                        'caption': caption
                    }
                    items.append(data)
except Exception as e:
    print(e)
    pass

import random
random.shuffle(items)
# print(items)
items_final = []
for idx in range(min(length, len(items))):
    items_final.append(items[idx])

with open('/data/Tsinghua/qinyf/projects/icme_finetune/json/train_sub.json', 'r') as f:
    interior58k_items = json.load(f)

random.shuffle(interior58k_items)

num = all_len - len(items_final)
assert num < 1000

for idx in range(num):
    try:
        Image.open(interior58k_items[idx]['image'])
        Image.open(interior58k_items[idx]['conditioning_image'])
        items_final.append(interior58k_items[idx])
    except Exception:
        continue
# assert len(items_final)>900
random.shuffle(items_final)
with open(f'outputs/train.json', 'w') as f:
    json.dump(items_final, f, indent=4)

with open("log.txt", "a", encoding="utf-8") as file:
    # 向文件中写入内容
    file.write(f"{image_dirs}----{num}\n")
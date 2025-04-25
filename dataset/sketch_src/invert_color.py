from PIL import Image
import os
from tqdm import tqdm
root_dir = '/data1/jiangzj/code/IAT/lineart'
files = os.listdir(root_dir)
for file in tqdm(files):
    # 打开一张图片
    image_path = os.path.join(root_dir, file)  # 替换为你的图片路径
    image = Image.open(image_path)

    # 将图片转换为RGB模式，如果已经是RGB则无需这一步
    image = image.convert('RGB')

    # 反色处理
    inverted_image = Image.eval(image, lambda p: 255 - p)

    # 显示反色后的图片
    inverted_image.show()

    # 保存反色后的图片
    inverted_image.save(os.path.join(root_dir,f'{file}_invert.png'))
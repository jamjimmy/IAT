
# tips

- 一定要找到便捷的vscode-终端路径剪贴/编辑/补全方案
- 保持记录配置文档的习惯
- 可以开始建立自己的常用python代码库，并且找到一个快速调用的解决方案

# data collecting 

source:

ls /Node11_nvme/jiangzj/scp-diff-toolkit_old/data/IAT-seg
ls /Node10_nvme/car-crawler/data

## link

interior:
/Node11_nvme/jiangzj/scp-diff-toolkit_old/data/IAT-seg/train/image
ln -s /Node11_nvme/jiangzj/scp-diff-toolkit_old/data/IAT-seg/test/image/* car_class/test/interior

appearance:
extract jpeg from:
/Node10_nvme/car-crawler/data/wheelsage_native/.../*.jpeg
ln -s /Node10_nvme/car-crawler/data/wheelsage_native/*/*.jpeg car_class/train/appearance
ln -s /Node10_nvme/car-crawler/data/wheelsage_native/*/*.jpeg car_class/test/appearance

too much ?

**note:** os.walk needs to add followlinks=True to continue go into links!

## meta

parser.add_argument('--dir', type=str, required=True, help="要遍历的目录")
parser.add_argument('--output', type=str, required=True, help="输出文件的名称")

python extract_img.py --dir car_class/train --output train.txt

# setup enviorionment

check cuda version

conda

# change method: use timm

## trouble shooting 1

[Python程序中PIL Image "image file is truncated"问题分析与解决_image file is truncated (35 bytes not processed)-CSDN博客](https://blog.csdn.net/scool_winter/article/details/89426509)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


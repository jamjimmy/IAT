import os

# 定义源文件路径和目标目录
source_file = 'IAT__interior.txt'
target_directory = '/gpfs/essfs/iat/Tsinghua/crawler_data/interior50K'  # 请替换为实际目标目录
if not os.path.exists(target_directory):  # 如果目标目录不存在
    os.makedirs(target_directory)  # 创建目标目录

# 读取源文件中的路径
with open(source_file, 'r', encoding='utf-8') as file:
    paths = file.readlines()

# 创建软连接
for path in paths:
    path = path.strip()  # 去除换行符
    if os.path.exists(path):  # 检查源文件是否存在
        # 获取相对路径
        relative_path = os.path.relpath(path, start='/gpfs/essfs/iat/Tsinghua/crawler_data')  # 获取相对路径
        target_path = os.path.join(target_directory, relative_path)  # 目标路径
        target_dir = os.path.dirname(target_path)  # 获取目标目录
        if not os.path.exists(target_dir):  # 如果目标目录不存在
            os.makedirs(target_dir)  # 创建目标目录
        os.symlink(path, target_path)  # 创建软连接
    else:
        print(f"源文件不存在: {path}")

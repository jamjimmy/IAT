import os

source_file_1 = 'IAT__interior.txt'
source_file_2 = 'IAT__appearance.txt'  # 请替换为实际的第二个文件路径
source_path = '/gpfs/essfs/iat/Tsinghua/crawler_data'
target_directory = '/gpfs/essfs/iat/Tsinghua/crawler_data/left'  # 请替换为实际目标目录
if not os.path.exists(target_directory):  # 如果目标目录不存在
    os.makedirs(target_directory)  # 创建目标目录


# 读取第一个源文件中的路径
with open(source_file_1, 'r', encoding='utf-8') as file:
    paths_1 = set(file.readlines())

# 读取第二个源文件中的路径
with open(source_file_2, 'r', encoding='utf-8') as file:
    paths_2 = set(file.readlines())

# 合并两个路径集合
all_paths = paths_1.union(paths_2)

# 遍历目标目录下的文件
for root, dirs, files in os.walk(source_path):
    for file in files:
        file_path = os.path.join(root, file)
        if file_path not in all_paths:  # 如果文件路径不在记录的路径中
            target_path = os.path.join(target_directory, file)  # 目标路径
            os.symlink(file_path, target_path)  # 创建软连接

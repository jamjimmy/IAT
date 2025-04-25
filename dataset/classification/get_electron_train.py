import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

def find_bottom_dirs(root_dir):
    """递归查找最底层目录"""
    bottom_dirs = []
    
    def dfs(current_dir):
        subdirs = [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]
        if not subdirs:  # 如果没有子目录，则为底层目录
            bottom_dirs.append(current_dir)
        else:
            for subdir in subdirs:
                dfs(os.path.join(current_dir, subdir))
    
    dfs(root_dir)
    return bottom_dirs

def is_before_2015(dir_name):
    """检查目录名是否表示2015年之前的年款"""
    try:
        # 提取年份部分（假设格式为"xxxx款"，其中xxxx是年份）
        year_str = os.path.basename(dir_name).split('款')[0]
        if year_str.isdigit() and int(year_str) < 2015:
            return True
    except:
        pass
    return False

def collect_image_paths(directories, target_count=1000):
    """收集指定目录下的所有图片路径"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    
    for directory in directories:
        if len(all_images) >= target_count:
            break
            
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    all_images.append(os.path.join(root, file))
                    if len(all_images) >= target_count:
                        break
            if len(all_images) >= target_count:
                break
    
    return all_images[:target_count]

def create_symlinks(image_paths, target_dir):
    """创建软链接到目标目录"""
    # 创建目标目录结构
    train_dir = os.path.join(target_dir, 'train')
    test_dir = os.path.join(target_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 分割训练集和测试集
    train_count = min(800, int(len(image_paths) * 0.8))
    train_images = image_paths[:train_count]
    test_images = image_paths[train_count:]
    
    # 创建软链接
    def create_link(src, dst_dir):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda img: create_link(img, train_dir), train_images)
        executor.map(lambda img: create_link(img, test_dir), test_images)
    
    print(f"已创建 {len(train_images)} 张训练图片和 {len(test_images)} 张测试图片的软链接")

def main(source_dir, target_dir):
    # 查找所有底层目录
    print("正在查找底层目录...")
    bottom_dirs = find_bottom_dirs(source_dir)
    
    # 筛选2015年之前的目录
    print("正在筛选2015年之前的目录...")
    old_car_dirs = [d for d in bottom_dirs if is_before_2015(d)]
    print(f"找到 {len(old_car_dirs)} 个2015年之前的车型目录")
    
    # 收集图片路径
    print("正在收集图片路径...")
    image_paths = collect_image_paths(old_car_dirs, 1000)
    print(f"收集了 {len(image_paths)} 张图片")
    
    # 随机打乱图片顺序
    random.shuffle(image_paths)
    
    # 创建软链接
    print("正在创建软链接...")
    create_symlinks(image_paths, target_dir)
    
    print("处理完成！")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="收集2015年前的汽车图片并创建软链接")
    parser.add_argument("--source", default="/gpfs/essfs/iat/Tsinghua/crawler_data/interior57K/train/autohome_interior", help="源目录路径")
    parser.add_argument("--target", default="carclass", help="目标目录路径")
    
    args = parser.parse_args()
    main(args.source, args.target)

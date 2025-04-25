import os
import argparse

def is_image_file(filename):
    """检查文件是否为图片格式"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def find_images(directory):
    """递归遍历目录，找到所有图片文件"""
    image_paths = []
    for root, _, files in os.walk(directory, followlinks=True):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(root, file))
    return image_paths

def write_to_file(image_paths, output_file):
    """将图片路径写入指定的txt文件"""
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(path + '\n')

def main():
    parser = argparse.ArgumentParser(description="递归遍历指定目录下所有图片，并将路径输出到指定的txt文件")
    parser.add_argument('--dir', type=str, required=True, help="要遍历的目录")
    parser.add_argument('--output', type=str, required=True, help="输出文件的名称")
    args = parser.parse_args()

    image_paths = find_images(args.dir)
    write_to_file(image_paths, args.output)
    print(f"已将 {len(image_paths)} 个图片路径写入到 {args.output}")

if __name__ == "__main__":
    main()
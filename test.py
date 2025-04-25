import json
import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="处理输入文件并生成 JSON 输出文件")
    parser.add_argument('-i', '--input', required=True, help="输入文件路径")
    parser.add_argument('-o', '--output', required=True, help="输出文件路径")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    try:
        result = []
        with open(input_file, 'r') as f:
            for idx, line in tqdm(enumerate(f.readlines())):
                if line.startswith("Prompt"):
                    prompt = line[12:].strip()
                    result.append({"caption": prompt})

        # 确保输出目录存在
        output_json_dir = os.path.dirname(output_file)
        os.makedirs(output_json_dir, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

        print(f"处理完成，结果已保存到 {output_file}")

    except FileNotFoundError:
        print(f"未找到输入文件，请检查文件路径: {input_file}")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main()
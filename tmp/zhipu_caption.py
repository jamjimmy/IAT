import torch
from diffusers import FluxPipeline
from accelerate import Accelerator
import json
import os
from tqdm import tqdm
import shutil
import uuid
from zhipuai import ZhipuAI
from accelerate.utils import gather_object
import re
import sys

def extract_double_quoted_content(input_string):
    # 使用正则表达式匹配双引号之间的内容
    pattern = r'"(.*?)"'
    # 找到所有匹配的内容
    matches = re.findall(pattern, input_string)
    return matches



def generate_json(totle_len=4):
        prompts = []
        idx = 0
        progress_bar = tqdm(total=totle_len, desc="Processing Zhipu")
        while(len(prompts)<totle_len):
            idx+=1
            if idx >200:
                exit()
            try:
                response = client.chat.completions.create(
                    model="glm-4-flash",  # 请填写您要调用的模型名称
                    messages=[
                        {"role": "user", "content": "帮我设计一辆现代的，充满科技感的汽车内饰设计。这里给你一个例子:'[Photograph of a futuristic,minimalist car interior at sunset. The car features a sleek,white and yellow leather dashboard,digital display,and a quilted patterned door trim. The spacious,modern design includes a central console with a gear shifter and cup holders. The background shows a serene lake and mountains under a golden sky.]'。请你发挥想象力，设计的尽量好看。用英文回复我。长度类似示例，并且不要有其他多余的东西。用python列表的形式返回给我5个。每个prompt用双引号包裹。"},
                    ],
                )
                text = response.choices[0].message.content
                # 调用函数提取双引号之间的内容
                result_list = extract_double_quoted_content(text)
                if len(result_list) <7:
                    for item in result_list:
                        prompts.append({"caption": item})
                        progress_bar.update(1)

                with open(f'{output_dir}/caption.json', 'w') as f:
                    json.dump(prompts, f, indent=4)


            except Exception as e:
                print(e)
                continue


        os.makedirs(f'{output_dir}/', exist_ok=True)
        with open(f'{output_dir}/caption.json', 'w') as f:
            json.dump(prompts, f, indent=4)
        return prompts

for i in range(50):
    total_len = 300
    client = ZhipuAI(api_key="c9bcc8c8eae1267231a9abc902d7c43d.PdJbzq90JNC8ngNT")  # 请填写您自己的APIKey
    output_dir = f"outputs/outputs_zhipu/{i}"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, 'caption.json')):
        continue
    if i == 0:
        prompts = generate_json(2*total_len)
    else:
        prompts = generate_json(total_len)

    with open('outputs/zhipu.json', 'r') as f:
        items = json.load(f)
    for item in prompts:
        items.append(item)
    with open('outputs/zhipu.json', 'w') as f:
        json.dump(items, f, indent=4)
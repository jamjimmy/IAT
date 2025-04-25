import torch
from diffusers import FluxPipeline
import json
import os
from tqdm import tqdm
import shutil
pipe = FluxPipeline.from_pretrained("/data/Tsinghua/Share/HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("./train_ckpt/INT_Realistic_20250301.safetensors")
pipe.fuse_lora(lora_scale=0.7)
pipe.to("cuda")

# TODO:需要替换这里为之前生成的prompt文件，根据生成的格式改下读写方式
with open('./json/4_10_1.json', 'r') as f:
    items = json.load(f)
for idx, item in tqdm(enumerate(items)):
    # TODO:修改这里，根据需要修改输出路径
    # output_dir_path = f"./output_4_4_2/{idx}"
    output_dir_path = f"./outputs/output_4_10_1/{idx}"
      
    if os.path.exists(output_dir_path):
        print(f"output dir {output_dir_path} already exists, skipping...")
        continue
    os.makedirs(output_dir_path, exist_ok=True)
    for i in range(10):
        prompt = item['caption']
        image = pipe(prompt, 
                    num_inference_steps=24, 
                    guidance_scale=3.5,
                    width=1024, height=768,
                    ).images[0]
        image.save(os.path.join(output_dir_path, f"{i}.png"))

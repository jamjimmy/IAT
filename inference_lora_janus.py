# # CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 accelerate launch --config_file /Node11_nvme/jiangzj/IAT/accelerate_lora.yaml --multi_gpu --mixed_precision bf16  --num_processes 21  inference_lora_jzj.py    
import torch
from diffusers import FluxPipeline
from accelerate import Accelerator
import json
import os
import sys
from tqdm import tqdm

# 初始化 Accelerator
accelerator = Accelerator()
device = accelerator.device

# 获取输出路径
output_dir = sys.argv[1]
os.makedirs(output_dir, exist_ok=True)

# 加载 caption.json
with open(os.path.join(output_dir, 'caption.json'), 'r') as f:
    items = json.load(f)

# 加载模型
pipe = FluxPipeline.from_pretrained(
    "/data/Tsinghua/Share/HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("./train_result/INT_Realistic_20250301.safetensors")
pipe.fuse_lora(lora_scale=0.7)
pipe.enable_model_cpu_offload(gpu_id=device.index)

# 正确使用 split_between_processes：不要加 `with`
local_items = accelerator.split_between_processes(items)
with accelerator.split_between_processes(items) as local_items:
    # 每个进程只处理自己的一部分数据
    for local_idx, item in enumerate(tqdm(local_items, disable=not accelerator.is_local_main_process)):
        global_idx = accelerator.process_index * len(local_items) + local_idx
        save_path = os.path.join(output_dir, str(global_idx))
        os.makedirs(save_path, exist_ok=True)

        # 保存 caption
        with open(os.path.join(save_path, "caption.txt"), "w") as f:
            f.write(item["caption"])

        # 保存图片
        image_path = os.path.join(save_path, "0.jpg")
        if os.path.exists(image_path):
            continue

        image = pipe(
            item["caption"],
            num_inference_steps=24,
            guidance_scale=3.5,
            width=1368, height=1024,
        ).images[0]
        image.save(image_path)

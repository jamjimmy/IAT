import torch
from diffusers import FluxPipeline
from accelerate import Accelerator
import json
import os
from tqdm import tqdm
import shutil
import uuid
from accelerate.utils import gather_object
def get_batches(items, batch_size=501):
    num_batches = (len(items) + batch_size - 1) // batch_size
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(items))
        batch = items[start_index:end_index]
        batches.append(batch)

    return batches

# 加载模型
pipe = FluxPipeline.from_pretrained(
    "/data/Tsinghua/Share/HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/",
    torch_dtype=torch.bfloat16
)

pipe.load_lora_weights("./train_result/INT_Realistic_20250301.safetensors")
pipe.fuse_lora(lora_scale=0.7)

distributed_state = Accelerator()
pipe.enable_model_cpu_offload(gpu_id=distributed_state.device.index)



# 读取 prompt
with open('../Janus/caption.json', 'r') as f:
    items = json.load(f)
# items.shuffle()
loader = get_batches(items, 1)

output_dir_path = "./outputs/output_50k"
if distributed_state.is_main_process:
    os.makedirs(output_dir_path, exist_ok=True)


# for _, data_raw in tqdm(enumerate(loader), total=len(loader)):
#     with distributed_state.split_between_processes(data_raw) as data:
#         result = pipe(prompt, 
#                   num_inference_steps=24, 
#                   guidance_scale=3.5,
#                   width=1024, height=768,
#                   ).images[0]
#         result.save(f"{output_dir_path}/result_{state.process_index}.png")

for idx, data_raw in tqdm(enumerate(loader), total=len(loader)):
    if os.path.exists(os.path.join(output_dir_path, str(idx))):
        # continue
        pass
    
    os.makedirs(os.path.join(output_dir_path, str(idx)), exist_ok=True)
    with open(os.path.join(output_dir_path, str(idx), 'caption.txt'), 'w') as f:
        f.write(data_raw[0]['caption']) 
    with distributed_state.split_between_processes(data_raw) as data:
        d = data[0]
        prompt = d['caption']
        # prompt="Car interior image, high definition, very realistic, with bright indoor lighting. The light-colored interior is elegant and dignified, exuding a strong sense of technology and futurism. It features metallic trim"
        for i in range(50):
            # try:
            pred = pipe(prompt, 
                  num_inference_steps=24, 
                  guidance_scale=3.5,
                  width=1024, height=768,
                  ).images[0]
            # pred.save(os.path.join(output_dir_path, str(idx),  str(uuid.uuid4())+'.jpg'))

        # save_items = []
        # with open(os.path.join(output_dir_path, 'caption.json'), 'a') as f:
        #     for image in os.listdir(os.path.join(output_dir_path, str(idx))):
        #         save_item = {
        #             "image": os.path.join(output_dir_path, str(idx), image),
        #             "caption": prompt
        #         }
        #         save_items.append(save_item)
        #     json.dump(save_item, f, ident=4)

            # distributed_state.wait_for_everyone()

            # preds = gather_object(pred)

            # if distributed_state.is_main_process:
            #     for pred in  preds:
                    

# prompts 是一个完整列表
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8 accelerate launch --config_file /Node11_nvme/jiangzj/IAT/accelerate_lora.yaml --multi_gpu --mixed_precision bf16  --num_processes 21  inference_lora_jzj.py    

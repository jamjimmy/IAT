import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from PIL import Image
import os
import json
import uuid

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='EXPERIMENT_1')
parser.add_argument('--controlnet_model', type=str, default='train_ckpt/flux_58k_controlnet')
# parser.add_argument('--start_num', type=int, default=0, required=True)
# parser.add_argument('--end_num', type=int, default=2000, required=True)
parser.add_argument('--controlnet_conditioning_scale', type=float, default=0.5)
parser.add_argument('--jsonl', type=str, required=True)
args = parser.parse_args()
controlnet_model = args.controlnet_model

input_dir = 'data/{}'.format(args.experiment_name)
output_dir = os.path.join('diffusers/result', args.experiment_name)
os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
base_model = '/data/Tsinghua/Share/HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, 
    controlnet=controlnet, 
    torch_dtype=torch.bfloat16
)
# enable memory optimizations   
pipe.enable_model_cpu_offload()
with open(args.jsonl, 'r') as f:
    items = json.load(f)
items.sort(key=lambda x: x['image'])
for idx, item in enumerate(items):
    img_path = item['conditioning_image']
    file_name = os.path.basename(img_path)
    if os.path.exists(os.path.join(output_dir, 'pred', file_name)) and os.path.exists(os.path.join(output_dir, 'gt', file_name)):
        continue
    prompt = item['caption']
    img = Image.open(img_path)
    control_image = load_image(img)
    size = img.size
    image = pipe(
        prompt, 
        control_image=control_image,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        num_inference_steps=28, 
        guidance_scale=3.5,
        width=1024,
        height=1024,
    ).images[0]
    # file_name, extension = os.path.splitext(os.path.basename(img_path))
    image.resize(size)
    image.save(os.path.join(output_dir, 'pred', file_name))
    os.symlink(item['image'], os.path.join(output_dir, 'gt', file_name))
    
# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from transformers import AutoModelForCausalLM
import os
import json

from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

# 输入的json路径
json_directory = 'electron.json'
with open(json_directory, 'r') as f:
    json_data = json.load(f)
    image_files = [item["image"] for item in json_data]

# 模型路径
model_path = "/gpfs/essfs/iat/Tsinghua/jiangzj/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()



# TODO: 这里的提问可能不一定会生成的很好，可以先跑一下看看怎么样，改一下问题
conversations = [
    [{
        "role": "User",
        "content": "<image_placeholder>\nGenerate a highly detailed description of a futuristic car interior in a photorealistic style, with a professional tone for a car design magazine. Place the interior in bright lighting conditions and visually striking environment.",
        "images": [image_file],
    },
    {"role": "Assistant", "content": ""},
    ]
    for image_file in image_files
]

results = []
for i, conversation in enumerate(conversations):
    for j in range(10):
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        results.append({
            "image": conversation[0]["images"][0],
            "caption": answer
        })

# TODO：将结果的列表保存在 JSON 文件
with open('caption.json', 'w') as f:
    json.dump(results, f, indent=4)

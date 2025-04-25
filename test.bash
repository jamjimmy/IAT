#!/bin/bash

# # 获取显卡信息
# nvidia_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)

# # 初始化变量
# max_free_memory=0
# best_gpu_index=-1

# # 逐行解析显卡信息
# while IFS=',' read -r index free_memory; do
#     if [ "$free_memory" -gt "$max_free_memory" ]; then
#         max_free_memory=$free_memory
#         best_gpu_index=$index
#     fi
# done <<< "$nvidia_info"

# # 输出结果
# if [ "$best_gpu_index" -ne -1 ]; then
#     echo "剩余显存最多的显卡索引是: $best_gpu_index，剩余显存: $max_free_memory MiB"
# else
#     echo "未找到可用的显卡。"
best_gpu_index=2
source /data/Tsinghua/new_anaconda3/etc/profile.d/conda.sh
cd /Node11_nvme/jiangzj/Janus
conda activate Comfyui-IAT
CUDA_VISIBLE_DEVICES=$best_gpu_index python /Node11_nvme/jiangzj/Janus/inference.py

# 检查 prompt_results.txt 是否为空
if [ ! -s /Node11_nvme/jiangzj/Janus/prompt_results.txt ]; then
    echo "prompt_results.txt 是空文件，脚本结束。"
    exit 1
fi

cp /Node11_nvme/jiangzj/Janus/prompt_results.txt /Node11_nvme/jiangzj/IAT/
cd /Node11_nvme/jiangzj/IAT/ 
cp prompt_results.txt ./txt/4_10_1.txt
# # cd ./txt
# # mv prompt_results.txt 4_10_1.txt
# # cd ..
python test.py -i ./txt/4_10_1.txt -o ./json/4_10_1.json

conda activate vsibench
CUDA_VISIBLE_DEVICES=$best_gpu_index python inference_lora.py

fi

# CUDA_VISIBLE_DEVICES=0,1,2,7 accelerate launch --config_file /Node11_nvme/jiangzj/IAT/accelerate_lora.yaml --multi_gpu --mixed_precision bf16  --num_processes 4 inference_lora_jzj.py 
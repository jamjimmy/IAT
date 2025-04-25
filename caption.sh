# #!/bin/bash
CUDA_VISIBLE_DEVICES=1  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=2  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=3  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=4  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=5  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=6  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=7  python Janus/inference_caption.py &
sleep 10 && CUDA_VISIBLE_DEVICES=0  python Janus/inference_caption.py &


# # 定义指令
# commands=(

# )

# # 启动指令
# for i in "${!commands[@]}"; do
#     echo "启动指令 $((i + 1))..."
#     eval "${commands[$i]}" &
#     sleep 20
# done

# # 等待所有后台任务完成
# wait
# echo "所有指令执行完成"
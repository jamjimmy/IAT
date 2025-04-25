export HF_ENDPOINT=https://hf-mirror.com
# export WANDB_MODE="offline"
export HF_HOME="./huggingface_home"
pretrain_path="../diffusers/FLUX.1-controlnet-lineart-promeai"
export WANDB_API_KEY="154d2536f85d4b2adc51e83c9dfccac4ca62d214"
# export TORCH_USE_CUDA_DSA
# accelerate  launch --main_process_port 62573 --config_file "./accelerate_config_zero3.yaml" --num_processes 1  train.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4 accelerate  launch  --main_process_port 15433 --config_file "./accelerate_config_zero3.yaml" --num_processes 5  train.py \
    --pretrained_model_name_or_path="/data/Tsinghua/Share/HF_HOME/hub/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44/" \
    --jsonl_for_train="json/new_50k_4000.json" \
    --conditioning_image_column=conditioning_image \
    --image_column=image \
    --caption_column=caption \
    --output_dir="train_result/new_50k" \
    --mixed_precision="bf16" \
    --resolution=512 \
    --learning_rate=1e-5 \
    --max_train_steps=6000 \
    --checkpointing_steps=1000 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --report_to="wandb" \
    --num_double_layers=4 \
    --num_single_layers=0 \
    --seed=42 \
    --controlnet_model_name_or_path="../diffusers/FLUX.1-controlnet-lineart-promeai" \
    --resume_from_checkpoint="latest"
#    --validation_steps=2 \
#    --validation_image="dataset/00001.jpg" \
#    --validation_prompt="The interior of this futuristic vehicle exudes a sense of luxury and innovation, designed to provide an unparalleled driving experience. The cabin is bathed in bright, natural light, highlighting the sleek and modern design elements that define this car. The seats are upholstered in a high-quality, light-colored leather, offering both comfort and a touch of elegance. The arrangement of the seats is spacious, with ample legroom and headroom, ensuring a comfortable ride for all passengers.\n\nThe dashboard is a masterpiece of modern technology, featuring a large, high-resolution touchscreen that seamlessly integrates with the vehicle's controls. This central display is flanked by advanced climate control systems and other essential functions, all easily accessible through intuitive touch interfaces. The steering wheel is ergonomically designed, with integrated controls for various functions, allowing the driver to maintain focus on the road while accessing important information and features.\n\nThe ambient lighting system is a standout feature, with soft, dynamic illumination that adapts to the mood and needs of the occupants. The overhead lighting is subtly integrated into the ceiling, creating a sense of openness and space. The vehicle's interior is further enhanced by subtle, high-tech accents, such as the use of advanced materials and subtle textures that add to the overall aesthetic appeal.\n\nOverall, this car interior is a testament to the future of automotive design, combining comfort, technology, and style in a way that is both functional and visually striking."
        

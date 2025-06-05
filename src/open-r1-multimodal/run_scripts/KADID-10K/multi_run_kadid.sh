cd src/open-r1-multimodal

export DEBUG_MODE="true"

RUN_NAME="KADID-N6-48-e10-Qwen2.5-VL-7B-GRPO"
export LOG_PATH="./log_$RUN_NAME.txt"

torchrun --nproc_per_node="8" \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path /home/notebook/data/group/wth/QualityReasoning/Models/Qwen2.5-VL-7B-Instruct \
    --dataset_name KADID-10K \
    --question_template scoring \
    --image_folders /home/notebook/data/group/wth/datasets/KADID-10K/images \
    --data_file_paths /home/notebook/data/group/wth/VisualQuality-R1/datasets/KADID-10K/scoring/RL-KADID-10K_train_scoring.jsonl \
    --freeze_vision_modules false \
    --max_prompt_length 1024 \
    --num_generations 6 \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to tensorboard \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 10 \
    --run_name $RUN_NAME \
    --save_steps 25 \
    --save_only_model true
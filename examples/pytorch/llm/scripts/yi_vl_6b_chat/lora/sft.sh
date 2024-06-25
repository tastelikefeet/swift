# Experimental environment: V100, A10, 3090
# 18GB GPU memory
NPROC_PER_NODE=4 \
swift sft \
    --model_type llava-qwen2 \
    --sft_type full \
    --template_type AUTO \
    --dtype AUTO \
    --output_dir output \
    --dataset llava-pretrain \
    --train_dataset_sample -1 \
    --num_train_epochs 2 \
    --max_length 4096 \
    --check_dataset_strategy warning \
    --gradient_checkpointing true \
    --batch_size 2 \
    --weight_decay 0.1 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 64 \
    --max_grad_norm 0.5 \
    --warmup_ratio 0.03 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 4 \
    --logging_steps 10 \
    --use_flash_attn true \
    --ddp_find_unused_parameters true

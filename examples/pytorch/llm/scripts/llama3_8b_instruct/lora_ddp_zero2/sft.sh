# Experimental environment: 2 * 3090
# 2 * 22GB GPU memory
nproc_per_node=4

NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model_type llama3-8b-instruct \
    --sft_type lora \
    --tuner_backend peft \
    --output_dir output \
    --num_train_epochs 1 \
    --max_length 2048 \
    --dataset toolbench_formatted \
    --check_dataset_strategy warning \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout_p 0.05 \
    --lora_target_modules ALL \
    --batch_size 2 \
    --weight_decay 0.1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --use_flash_attn true \
    --use_loss_scale true

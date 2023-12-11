PYTHONPATH=../../../ \
accelerate launch train_lcm_distill_sd_wds.py \
    --pretrained_teacher_model="AI-ModelScope/stable-diffusion-v1-5" \
    --output_dir="train_lcm_distill_sd_wds" \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="AI-ModelScope/conceptual-captions-12m-webdataset" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=tensorboard \
    --seed=453645634
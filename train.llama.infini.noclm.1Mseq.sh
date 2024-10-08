# export CUDA_VISIBLE_DEVICES=0

DEBUG=true 
# accelerate launch --mixed_precision='bf16' \
    # --num_processes=1 \
accelerate launch \
    --multi_gpu \
    --mixed_precision='bf16' \
    train.llama.infini.noclm.py \
    --model_name_or_path='meta-llama/Meta-Llama-3-8B' \
    --dataset_name='JeanKaddour/minipile' \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --output_dir='./models/llama-3-8b-infini-noclm-minipile' \
    --checkpointing_steps=1000 \
    --num_train_epochs=1 \
    --learning_rate=1e-4 \
    --seed=42 \
    --low_cpu_mem_usage \
    --report_to='wandb' \
    --preprocessing_num_workers=64 \
    --with_tracking \
    --gradient_accumulation_steps 2048 \
    --segment_length=2048 \
    --block_size=32768 \
    # --block_size=1048576 \

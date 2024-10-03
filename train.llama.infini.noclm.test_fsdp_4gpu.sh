# export CUDA_VISIBLE_DEVICES=0

echo "Running LLAMA3-8B INFINI NOCLM with FSDP 4-GPU on WIKITEXT"

export NCCL_DEBUG=INFO
export DEBUG=True
SEGMENT_LENGTH=2048  # note must match the model config?

accelerate launch \
    --mixed_precision='bf16' \
    --config_file="./accel_fsdp_4gpu.yaml" \
    train.llama.infini.noclm.py \
    --model_name_or_path='meta-llama/Meta-Llama-3-8B' \
    --segment_length=$SEGMENT_LENGTH \
    --block_size=4096 \
    --dataset_name='wikitext' \
    --dataset_config_name='wikitext-2-raw-v1' \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --weight_decay=1.0 \
    --output_dir='./models/llama3-8b-infini-noclm-wikitext' \
    --checkpointing_steps=10 \
    --num_train_epochs=1 \
    --learning_rate=5e-5 \
    --seed=42 \
    --report_to='wandb' \
    --preprocessing_num_workers=64 \
    --with_tracking \
    --gradient_accumulation_steps=16 \
    # --low_cpu_mem_usage \

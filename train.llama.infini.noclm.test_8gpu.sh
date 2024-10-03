# export CUDA_VISIBLE_DEVICES=0
# export NCCL_DEBUG=INFO
# export DEBUG=True 

accelerate launch \
    --mixed_precision='bf16' \
    --multi-gpu \
    --num_machines 1 \
    --num_processes 8 \
    --debug \
    test_train.multi.llama.infini.py


# accelerate launch \
#     --num_processes=1 \
#     --mixed_precision='bf16' \
    # train.llama.infini.noclm.py \
    # --model_name_or_path='meta-llama/Meta-Llama-3-8B' \
    # --segment_length=2048 \
    # --block_size=4096 \
    # --dataset_name='wikitext' \
    # --dataset_config_name='wikitext-2-raw-v1' \
    # --per_device_train_batch_size=2 \
    # --per_device_eval_batch_size=2 \
    # --output_dir='./models/llama-3-8b-infini-noclm-wikitext-test' \
    # --checkpointing_steps=10 \
    # --num_train_epochs=1 \
    # --learning_rate=5e-5 \
    # --seed=42 \
    # --low_cpu_mem_usage \
    # --report_to='wandb' \
    # --preprocessing_num_workers=64 \
    # --with_tracking

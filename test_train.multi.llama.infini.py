import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # TODO: set the GPU device
os.environ["WANDB_PROJECT"] = "InfiniTransformer"
# os.environ["WANDB_MODE"] = "offline"

# os.environ["DEBUG"] = "True"

# os.environ['DEEPSPEED_LOG_LEVEL'] = 'trace'  # For detailed logging

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


from itertools import chain

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)

from infini_llama import LlamaForCausalLM, LlamaConfig

# from accelerate import Accelerator

# accelerator = Accelerator()

import deepspeed

MODEL = "meta-llama/Meta-Llama-3-8B"

set_seed(42)

print("Torch Version:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

# if torch.cuda.is_available():
#     # device = "cuda:0"  # set GPU device using CUDA_VISIBLE_DEVICES
# else:
#     device = "cpu"
# device = accelerator.device
device = "cuda"
print(f"Device: {device}")


training_args = TrainingArguments(
    output_dir="./models/llama3-8b-wikitext",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # to test batch dim
    save_total_limit=1,
    report_to="wandb",  # "none" if you don't want to report to wandb
    run_name="llama3-8b-wikitext",
    optim="adafactor",
    learning_rate=1e-4,
    bf16=True,
    logging_first_step=True,
    logging_steps=1,
    save_strategy="epoch",
    # warmup_ratio=0.1,
    max_grad_norm=1.0,
    gradient_checkpointing=True,  # Reduce vram 69G -> 43G
    # gradient_accumulation_steps=1,
    # deepspeed="./zero3.json",
)

with deepspeed.zero.Init():
    if os.path.exists("./models/llama3-8b"):
        model = LlamaForCausalLM.from_pretrained(
            "./models/llama3-8b", torch_dtype="auto",
            # device_map="auto"
        )
        config = model.config
        print(config)
        print(model)
    else:
        config = LlamaConfig.from_pretrained(
            MODEL,
            attn_implementation="eager",
        )
        # config.max_position_embeddings = 128
        config.use_cache = False
        config.segment_size = config.max_position_embeddings

        print(config)

        # Create the Llama model with Infini-attention
        model = LlamaForCausalLM(config)
        # model = model.from_pretrained("meta-llama/Meta-Llama-3-8B")
        pretrained_model = LlamaForCausalLM.from_pretrained(
            MODEL, torch_dtype="auto"
        )
        # Step 4: Transfer weights
        # Note: This is a simplified example; you need to ensure that each parameter's dimensions match.
        for param in model.named_parameters():
            name = param[0]
            if name in pretrained_model.state_dict():
                # Check if dimensions match, and only then assign the weights
                if param[1].size() == pretrained_model.state_dict()[name].size():
                    param[1].data = pretrained_model.state_dict()[name].data.clone()
                else:
                    print(f"Skipping {name} due to size mismatch.")
        del pretrained_model
        print(model)
        model = model.to(torch.bfloat16)
        model = model.to(device)

# wiki = load_dataset("wikipedia", "20220301.en", split="train[:20000]")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize_function(examples):
    return tokenizer(examples["text"])


try:
    column_names = list(wiki["train"].features)
except KeyError:
    column_names = list(wiki.features)
tokenized_datasets = wiki.map(
    tokenize_function, remove_columns=column_names, batched=True
)


block_size = config.segment_size * 4  # will be 32768
print("block_size:", block_size)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
)

print(lm_datasets)
# print(lm_datasets["train"]["input_ids"][0])


try:
    train_dataset = lm_datasets["train"]
except KeyError:
    train_dataset = lm_datasets


# model, train_dataset, tokenizer = accelerator.prepare(
#     model, train_dataset, tokenizer
# )

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    # eval_dataset=lm_datasets["validation"],
    data_collator=default_data_collator,
)

trainer.train()


"""
   "optimizer": {
        "type": "ZeroOneAdam",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "bias_correction": false,
            "var_freeze_step": 1000,
            "var_update_scaler": 16,
            "local_step_scaler": 1000,
            "local_step_clipper": 16,
            "cuda_aware": false,
            "comm_backend_name": "nccl"
        }
    },
"""
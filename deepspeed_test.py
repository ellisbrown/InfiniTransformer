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
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)

from transformers import LlamaForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig


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

dschf = HfDeepSpeedConfig("./zero3.json")


training_args = TrainingArguments(
    output_dir="./models/llama3-8b-TEST",
    overwrite_output_dir=True,
    num_train_epochs=1,
    save_total_limit=1,
    report_to="wandb",  # "none" if you don't want to report to wandb
    run_name="llama3-8b-TEST",
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

# with deepspeed.zero.Init(
#     config_dict_or_path="./zero3.json",
#     enabled=True,
# ):

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
)
print(model)
print(f"Model #Params: {model.num_parameters()}")
model = model.to(torch.bfloat16)
model = model.to(device)


# wiki = load_dataset("wikipedia", "20220301.en", split="train[:20000]")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    # return tokenizer(examples["text"])
    result = tokenizer(examples["text"])
    result["labels"] = result["input_ids"].copy()
    del result["input_ids"]
    return result


try:
    column_names = list(wiki["train"].features)
except KeyError:
    column_names = list(wiki.features)

tokenized_datasets = wiki.map(
    tokenize_function, remove_columns=column_names, batched=True
)



try:
    train_dataset = tokenized_datasets["train"]
except KeyError:
    train_dataset = tokenized_datasets


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

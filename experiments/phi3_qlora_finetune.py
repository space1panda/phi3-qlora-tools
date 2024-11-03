from datetime import datetime

import bitsandbytes as bnb
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl import SFTTrainer

base_model_id = "microsoft/Phi-3.5-mini-instruct"


### DATA PREPARATION

dataset_name = 'Amod/mental_health_counseling_conversations'
dataset = load_dataset(dataset_name, split="train")


tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    model_max_length=1024,
    padding_side="left",
    add_eos_token=True)

tokenizer.pad_token = tokenizer.eos_token


def generate_prompt(data_point):
    """
    Generates a formatted prompt for fine-tuning from a data point.
    """
    prefix_text = "This is a conversation from a mental health therapy chat. Respond empathetically and informatively." #instruction
    context = data_point['Context']
    response = data_point['Response']
    formatted_prompt = f"<s>[INST] {prefix_text} {context} [/INST]{response}</s>"
    return formatted_prompt


# Assuming `dataset` is loaded using the datasets library
dataset = dataset.map(lambda x: {"prompt": generate_prompt(x)})


def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
dataset = tokenized_dataset.shuffle(seed=1234)


# Split the dataset into training and testing sets
train_test_split = dataset.train_test_split(test_size=0.05)  # 5% for testing, 95% for training
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

### MODEL Configuration

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config)

# base_model.gradient_checkpointing_enable()


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
    return list(lora_module_names)


modules = find_all_linear_names(base_model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)


trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total * 100:.4f}%")

project = "SFT-microsoft/Phi-3.5-mini-instruct"
base_model_name = "microsoft/Phi-3.5-mini-instruct"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name


tokenizer.pad_token = tokenizer.eos_token
torch.cuda.empty_cache()
model.training = True

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="prompt",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_checkpointing=False,
        # gradient_accumulation_steps=0,
        bf16=True,
        warmup_steps=5,
        max_steps=2000,
        learning_rate=2e-4,
        logging_steps=10,
        output_dir="outputs",
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=100,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        report_to="wandb",
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

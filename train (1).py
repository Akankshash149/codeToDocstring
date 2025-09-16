import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling, TrainerCallback
import sys, logging

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "qwen25-docgen-lora"


CURRENT_TRAIN_CAPACITY = 10000

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None: # if pad token is not define, use the end of segment token instead
    tokenizer.pad_token = tokenizer.eos_token

# loading the tokenized data from local drive
tokenized = load_from_disk("tokenized_docgen")

# shringking it for the current train cap
tokenized["train"] = tokenized["train"].select(range(CURRENT_TRAIN_CAPACITY))


device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
)
model.to(device)


# IMPORTANT -> LORA CONFIGURATION DETAILS 
lora_cfg = LoraConfig(
    r=4, # normally this is 16 or 8, but i have chosen 4 so as to reduce the work load 
    
    # lora alpha and dropout have the standard value which they are normally used with 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # targetting every module of the transformer
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   
        "gate_proj", "up_proj", "down_proj"       
    ]
)
# freezing the base model and adding the Lora adapter matrixes
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False   # for qwen we disable this
)

# for logging the details every 25 steps
class printFineTuningData(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"-> Step Number {state.global_step} - Logs: {logs}")
        sys.stdout.flush()
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("[*] Starting Model Fine Tuning : \n")
        sys.stdout.flush()

# IMPORTANT -> Training args 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # these 4 hyper params are subject to change, depending on how well the initial partial training of the model goes with these values
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    # #
    fp16=True,
    save_strategy="epoch",
    #eval_strategy="steps",      
    #eval_steps=500,
    dataloader_pin_memory=False,
    logging_first_step=True,
    logging_strategy="steps",   
    logging_steps=25,
    report_to=[],
)


# feeding in all the defined arguments into the trainer : 
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=collator,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],  
    callbacks=[printFineTuningData()]
)

trainer.train()

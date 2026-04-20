import torchvision
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig, logging

from datasets import Dataset

import csv, math, evaluate
import numpy as np

text = []
labels = []

dataset_csv = 'C:/Scripts/ExternalComments.csv'

with open(dataset_csv, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        labels.append(int(row[0]))
        text.append(row[1])

split = 0.7
split_index = math.floor(len(labels) * split)

split_dataset = {}
split_dataset["train"] = {"labels": labels[0:split_index], "text": text[0:split_index]}
split_dataset["test"] = {"labels": labels[split_index+1:-1], "text": text[split_index+1:-1]}


id2label = {0: "BAD",1: "GOOD"}
label2id = {"BAD": 0, "GOOD": 1}

print ("About to load")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", attn_implementation="flash_attention_2", truncation=True, max_length=1024)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # For Classification, usually padding on the left is better for Gemma/Llama
    tokenizer.padding_side = "left" 

print ("Tokenizer Loaded")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForSequenceClassification.from_pretrained("google/gemma-3-4b-it",
                                                           quantization_config=quantization_config,
                                                           num_labels=2, 
                                                           id2label=id2label, 
                                                           label2id=label2id, 
                                                           device_map="cuda:0")
model.config.pad_token_id = tokenizer.pad_token_id
model.gradient_checkpointing_enable()

print ("loading PEFT and LoRA")

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target key layers
    lora_dropout=0.05, 
    bias="none", 
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)

print ("Sequence Classifier Loaded")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

print ("tokenizing training data")

def preprocess_function(data):
    tk = tokenizer(data["text"], truncation=True)
    tk["token_type_ids"] = [[0] * len(input_id_seq) for input_id_seq in tk["input_ids"]]
    return tk

formatted_dataset_test = Dataset.from_dict(split_dataset["test"])
tokenized_data_test = formatted_dataset_test.map(preprocess_function, batched=True, remove_columns="text")

formatted_dataset_train = Dataset.from_dict(split_dataset["train"])
tokenized_data_train = formatted_dataset_train.map(preprocess_function, batched=True, remove_columns="text")

print ("data tokenized")

training_args = TrainingArguments(
    output_dir="ExternalCommentsModel",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="no",
    push_to_hub=False,
    optim="paged_adamw_32bit",
    bf16=True, 
    logging_steps=1,
    dataloader_num_workers=0
)

print ("declaring trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data_train,
    eval_dataset=tokenized_data_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print ("beginning training")

logging.enable_progress_bar()
logging.set_verbosity_info()

trainer.train()

# print ("training complete")
# trainer.push_to_hub()
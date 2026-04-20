import torchvision
import torch
from transformers import AutoTokenizer

import csv, math
import numpy as np

dataset_csv = 'C:/Scripts/ExternalComments.csv'

dataset = []
with open(dataset_csv, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        dataset.append({"text":row[1],"label": int(row[0])})

print("launching tokenizer")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

total = len(dataset)
print("total number of items = " + str(total))

i = 1
for dataset_entry in dataset:
    print (str(i) + " / " + str(total))
    dataset_entry["tokenized"] = (tokenizer(dataset_entry["text"]))
    i += 1

keys = dataset[0].keys()

with open('tokenized_dataset.csv', 'w', newline='') as csvfile:
    w = csv.DictWriter(csvfile,keys)
    w.writerows(dataset)

# split = 0.7
# split_index = math.floor(len(dataset) * split)

# split_dataset = {}
# split_dataset["train"] = dataset[0:split_index]
# split_dataset["test"] = dataset[split_index+1:-1]



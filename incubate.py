import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import re
import json
from tqdm import tqdm

from collections import Counter

from classifier import Classifier

import argparse

parser = argparse.ArgumentParser(description='Parser for Incubator.')

parser.add_argument('--n_epoch', type=int)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--device', type=int)
parser.add_argument('--n_sample', type=int)
parser.add_argument('--max_new_tokens', type=int)
parser.add_argument('--nli_finetune_epoch', type=int)
parser.add_argument('--instruction', type=str)
parser.add_argument('--incubator', type=str)
parser.add_argument('--classifier', type=str)
parser.add_argument('--save_path', type=str)

args = parser.parse_args()

n_epoch = args.n_epoch
batch_size = args.batch_size
device = args.device
n_sample = args.n_sample
max_new_tokens = args.max_new_tokens
instruction = args.instruction
incubator = args.incubator
classifier = args.classifier
save_path = args.save_path

tokenizer = AutoTokenizer.from_pretrained(incubator)
model = AutoModelForCausalLM.from_pretrained(incubator, torch_dtype=torch.float16)
model = model.to(f"cuda:{device}")

input_text = f"[INST] {instruction} [/INST]"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:1")

dataset = []

with torch.no_grad():
    for _ in tqdm(range(n_sample)):
        try:
            outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens, do_sample=True)
            data = re.findall("({.*?})", tokenizer.decode(outputs[0]))[0]
            data = json.loads(data)
            dataset.append(data)
        except:
            pass

labels = ["#".join(list(data.keys())) for data in dataset]

label_texts = Counter(labels).most_common(1)[0][0].split("#")

new_dataset = []

for data in dataset:
    if list(data.keys()) == label_texts:
        for label in data:
            new_dataset.append({"text": data[label], "label": label})
            
classifier = Classifier(model_name=classifier, device=f"cuda:{device}", num_labels=len(label_texts), label_texts=label_texts)

for epoch in range(n_epoch):
    classifier.train(new_dataset, batch_size)
    
classifier.tok.save_pretrained(save_path)
classifier.classifier.save_pretrained(save_path)
    
for input_text in ["I love 'Spiderman 2'!", "I ate a delicious pudding!"]:
    label = label_texts[classifier.predict(input_text).argmax().item()]
    print("Input:", input_text)
    print("Predicted Label:", label)
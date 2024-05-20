import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import re
import json
from tqdm import tqdm

from collections import Counter

from classifier import Classifier

n_epoch = 16
batch_size = 4
device = 1
n_sample = 16
max_new_tokens = 64
instruction = "Build a classifier that can categorize text messages by 'about food' and 'about movie'."
model_id = "KomeijiForce/Incubator-llama-2-7b"
classifier = "roberta-base"
save_path = "roberta-base-incubated"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
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
import torch
from torch import nn
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from tqdm import tqdm

class Classifier:

    def __init__(self, model_name='roberta-base', device='cuda:0', num_labels=2, learning_rate=1e-5, eps=1e-6, betas=(0.9, 0.999), label_texts=None):
        self.label_texts = label_texts
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = Adam([p for p in self.classifier.parameters()], lr=learning_rate, eps=eps, betas=betas)

    def train(self, dataset, batch_size=16):
    
        bar = tqdm(range(0, len(dataset), batch_size), leave=False)

        for idx in bar:
            tups = dataset[idx:idx + batch_size]
            texts = [tup["text"] for tup in tups]
            golds = [self.label_texts.index(tup["label"]) for tup in tups]

            inputs = self.tok(texts, padding=True, return_tensors='pt').to(self.device)
            scores = self.classifier(**inputs)[-1]
            golds = torch.LongTensor(golds).to(self.device)

            self.classifier.zero_grad()

            loss = self.criterion(scores, golds).mean()

            loss.backward()

            self.optimizer.step()

            bar.set_description(f'@Train #Loss={loss:.4}')

    def predict(self, text):
        
        inputs = self.tok(text, padding=True, return_tensors='pt').to(self.device)
        scores = self.classifier(**inputs).logits
                
        return scores[0]
"""
89. 事前学習済み言語モデルからの転移学習
事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
"""

import torch
from transformers import BertForSequenceClassification, BertTokenizer

# モデルとトークナイザーのロード
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_categories)
tokenizer = BertTokenizer.from_pretrained(model_name)

# データの前処理
def preprocess(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# モデルのファインチューニング
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs = preprocess(batch['text'])
        labels = batch['label']
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

from sklearn.metrics import accuracy_score, classification_report

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        inputs = preprocess(batch['text'])
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds)
        true_labels.extend(batch['label'])

accuracy = accuracy_score(true_labels, predictions)
report = classification_report(true_labels, predictions)

print(f"Accuracy: {accuracy}")
print(report)

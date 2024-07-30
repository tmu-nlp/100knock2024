'''
89. 事前学習済み言語モデルからの転移学習Permalink
事前学習済み言語モデル（例えばBERTなど）を出発点として，
ニュース記事見出しをカテゴリに分類するモデルを構築せよ．
'''
#事前学習済み言語モデル（例えばBERTなど）を出発点として，ニュース記事見出しをカテゴリに分類するモデルを構築せよ．

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.preprocessing import LabelEncoder
from load_and_create_dict import *


#hyper params
epochs = 3

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, tokenizer, max_len):
    tokens = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"]

# tokenize
X_train, mask_train = tokenize_texts(train['TITLE'], tokenizer, max_len=15)
X_valid, mask_valid = tokenize_texts(valid['TITLE'], tokenizer, max_len=15)
X_test, mask_test = tokenize_texts(test['TITLE'], tokenizer, max_len=15)

# label encode
label_encoder = LabelEncoder()
y_train = torch.tensor(label_encoder.fit_transform(train['CATEGORY']), dtype=torch.long)
y_valid = torch.tensor(label_encoder.transform(valid['CATEGORY']), dtype=torch.long)
y_test = torch.tensor(label_encoder.transform(test['CATEGORY']), dtype=torch.long)

class BERTClass(torch.nn.Module):
    def __init__(self, drop_rate=0.04, output_size=4):
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = torch.nn.Linear(768, output_size)  # BERTの出力に合わせて768次元を指定

    def forward(self, ids, mask):
        ret = self.bert_model(input_ids=ids, attention_mask=mask)
        last_hidden_state = ret["last_hidden_state"]
        x = last_hidden_state[:, 0, :]
        logit = self.fc(x)
        return logit

#model
model = BERTClass(output_size=len(label_encoder.classes_))

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#data loader
train_dataset = TensorDataset(X_train.to(device), mask_train.to(device), y_train.to(device))
valid_dataset = TensorDataset(X_valid.to(device), mask_valid.to(device), y_valid.to(device))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

#train model
for epoch in range(epochs):
    model.train()
    total_loss_train = 0
    total_correct_train = 0

    for ids, mask, labels in train_loader:
        optimizer.zero_grad()
        y_pred = model(ids, mask)
        loss = loss_fn(y_pred, labels)
        loss.backward()
        optimizer.step()
        
        total_loss_train += loss.item()
        total_correct_train += (y_pred.argmax(dim=1) == labels).sum().item()

    model.eval()
    total_loss_valid = 0
    total_correct_valid = 0

    with torch.no_grad():
        for ids, mask, labels in valid_loader:
            y_pred = model(ids, mask)
            loss = loss_fn(y_pred, labels)
            total_loss_valid += loss.item()
            total_correct_valid += (y_pred.argmax(dim=1) == labels).sum().item()

    print(f"Epoch: {epoch + 1}")
    print(f"Train Loss: {total_loss_train / len(train_loader.dataset):.4f}, Train Acc: {total_correct_train / len(train_loader.dataset):.4f}")
    print(f"Valid Loss: {total_loss_valid / len(valid_loader.dataset):.4f}, Valid Acc: {total_correct_valid / len(valid_loader.dataset):.4f}")

"""
output:
Epoch: 1
Train Loss: 0.0106, Train Acc: 0.8854
Valid Loss: 0.0077, Valid Acc: 0.9214
Epoch: 2
Train Loss: 0.0046, Train Acc: 0.9518
Valid Loss: 0.0072, Valid Acc: 0.9304
Epoch: 3
Train Loss: 0.0024, Train Acc: 0.9764
Valid Loss: 0.0087, Valid Acc: 0.9222
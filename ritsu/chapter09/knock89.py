import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
<<<<<<< HEAD
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
>>>>>>> 42dbd97f744e68a9651be100bfd37f30a9607416
=======
import matplotlib.pyplot as plt
>>>>>>> eba84b5f99df32efcb379942275cdda8e709fc87

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データセットクラスの定義
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# モデルクラスの定義
class BertClassifier(nn.Module):
    def __init__(self, n_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)
        return self.out(output)

<<<<<<< HEAD
<<<<<<< HEAD
# データの読み込みと前処理
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['CATEGORY', 'TITLE'])
    return df

base_path = os.path.join('..', 'chapter06')
train_data = load_data(os.path.join(base_path, 'train.txt'))
valid_data = load_data(os.path.join(base_path, 'valid.txt'))
test_data = load_data(os.path.join(base_path, 'test.txt'))

# カテゴリをインデックスに変換
category_to_index = {'b': 0, 't': 1, 'e': 2, 'm': 3}
train_data['CATEGORY'] = train_data['CATEGORY'].map(category_to_index)
valid_data['CATEGORY'] = valid_data['CATEGORY'].map(category_to_index)
test_data['CATEGORY'] = test_data['CATEGORY'].map(category_to_index)

# BERTトークナイザーの初期化
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# データセットの作成
MAX_LEN = 128
BATCH_SIZE = 16

train_dataset = NewsDataset(
    texts=train_data.TITLE.to_numpy(),
    labels=train_data.CATEGORY.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

valid_dataset = NewsDataset(
    texts=valid_data.TITLE.to_numpy(),
    labels=valid_data.CATEGORY.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = NewsDataset(
    texts=test_data.TITLE.to_numpy(),
    labels=test_data.CATEGORY.to_numpy(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2
)

# モデルの初期化
model = BertClassifier(n_classes=4)
model = model.to(device)

# 損失関数とオプティマイザの定義
EPOCHS = 10
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
total_steps = len(train_data_loader) * EPOCHS
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
loss_fn = nn.CrossEntropyLoss().to(device)

=======
>>>>>>> 42dbd97f744e68a9651be100bfd37f30a9607416
=======
>>>>>>> eba84b5f99df32efcb379942275cdda8e709fc87
# 訓練関数
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for batch in tqdm(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# 評価関数
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

<<<<<<< HEAD
<<<<<<< HEAD
# 訓練ループ
best_accuracy = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    
    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_data)
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(
        model,
        valid_data_loader,
        loss_fn,
        device,
        len(valid_data)
    )
    
    print(f'Val loss {val_loss} accuracy {val_acc}')
    print()
    
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

# テストデータでの評価
model.load_state_dict(torch.load('best_model_state.bin'))
test_acc, _ = eval_model(
    model,
    test_data_loader,
    loss_fn,
    device,
    len(test_data)
)

print(f'Test accuracy: {test_acc.item()}')
=======
=======
>>>>>>> eba84b5f99df32efcb379942275cdda8e709fc87
# データの読み込みと前処理
def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t', names=['CATEGORY', 'TITLE'])
    return df

# メイン処理
def main():
    base_path = os.path.join(os.path.dirname(__file__), '..', 'chapter06')
    train_data = load_data(os.path.join(base_path, 'train.txt'))
    valid_data = load_data(os.path.join(base_path, 'valid.txt'))
    test_data = load_data(os.path.join(base_path, 'test.txt'))

    # カテゴリをインデックスに変換
    category_to_index = {'b': 0, 't': 1, 'e': 2, 'm': 3}
    train_data['CATEGORY'] = train_data['CATEGORY'].map(category_to_index)
    valid_data['CATEGORY'] = valid_data['CATEGORY'].map(category_to_index)
    test_data['CATEGORY'] = test_data['CATEGORY'].map(category_to_index)

    # BERTトークナイザーの初期化
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # データセットの作成
    MAX_LEN = 128
    BATCH_SIZE = 16

    train_dataset = NewsDataset(
        texts=train_data.TITLE.to_numpy(),
        labels=train_data.CATEGORY.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    valid_dataset = NewsDataset(
        texts=valid_data.TITLE.to_numpy(),
        labels=valid_data.CATEGORY.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    test_dataset = NewsDataset(
        texts=test_data.TITLE.to_numpy(),
        labels=test_data.CATEGORY.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    # モデルの初期化
    model = BertClassifier(n_classes=4)
    model = model.to(device)

    # 損失関数とオプティマイザの定義
    EPOCHS = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # 訓練ループ
    best_accuracy = 0

    # 損失と精度を記録するリストを作成
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(train_data)
        )
        
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss = eval_model(
            model,
            valid_data_loader,
            loss_fn,
            device,
            len(valid_data)
        )
        
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()
        
        # 損失と精度を記録
        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item())
        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())
        
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    # テストデータでの評価
    model.load_state_dict(torch.load('best_model_state.bin'))
    test_acc, _ = eval_model(
        model,
        test_data_loader,
        loss_fn,
        device,
        len(test_data)
    )

    print(f'Test accuracy: {test_acc.item()}')

    # プロットの作成
    plt.figure(figsize=(12, 8))

    # 損失のプロット
    plt.subplot(2, 1, 1)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # 精度のプロット
    plt.subplot(2, 1, 2)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()

    # 画像の保存
    plt.savefig('knock89_image_loss_accuracy.png')
    plt.close()

    print("Loss and accuracy plots have been saved as 'knock89_image_loss_accuracy.png'")

if __name__ == '__main__':
    main()
<<<<<<< HEAD
>>>>>>> 42dbd97f744e68a9651be100bfd37f30a9607416
=======
>>>>>>> eba84b5f99df32efcb379942275cdda8e709fc87

"""
(以下はcolabでの実行結果)

Epoch 1/10
----------
100% 667/667 [00:55<00:00, 11.96it/s]
Train loss 0.45799246526182236 accuracy 0.8464205397301349
Val loss 0.2547949694673575 accuracy 0.9175412293853074

Epoch 2/10
----------
100% 667/667 [00:54<00:00, 12.22it/s]
Train loss 0.20129669322792826 accuracy 0.940123688155922
Val loss 0.22853561192529187 accuracy 0.9310344827586207

Epoch 3/10
----------
100% 667/667 [00:54<00:00, 12.23it/s]
Train loss 0.13773046938543287 accuracy 0.9627061469265367
Val loss 0.23280278574453578 accuracy 0.9415292353823088

Epoch 4/10
----------
100% 667/667 [00:54<00:00, 12.22it/s]
Train loss 0.0889263710703818 accuracy 0.976667916041979
Val loss 0.3145528290936324 accuracy 0.9370314842578711

Epoch 5/10
----------
100% 667/667 [00:54<00:00, 12.22it/s]
Train loss 0.056528849873256855 accuracy 0.9871626686656672
Val loss 0.28064170343055767 accuracy 0.9437781109445277

Epoch 6/10
----------
100% 667/667 [00:54<00:00, 12.24it/s]
Train loss 0.04213725335379739 accuracy 0.9904422788605697
Val loss 0.35647490077272304 accuracy 0.9385307346326837

Epoch 7/10
----------
100% 667/667 [00:54<00:00, 12.25it/s]
Train loss 0.038524589851352695 accuracy 0.9920352323838081
Val loss 0.3832121642569421 accuracy 0.93928035982009

Epoch 8/10
----------
100% 667/667 [00:54<00:00, 12.24it/s]
Train loss 0.02752914756118558 accuracy 0.9940029985007496
Val loss 0.4033368652453646 accuracy 0.9377811094452774

Epoch 9/10
----------
100% 667/667 [00:54<00:00, 12.23it/s]
Train loss 0.02694159595410457 accuracy 0.9939092953523238
Val loss 0.44699470007288344 accuracy 0.93928035982009

Epoch 10/10
----------
100% 667/667 [00:54<00:00, 12.24it/s]
Train loss 0.029474284924049073 accuracy 0.9939092953523238
Val loss 0.45648470134949004 accuracy 0.9355322338830584

<<<<<<< HEAD
<<<<<<< HEAD
Figure(1200x400)
=======
>>>>>>> 42dbd97f744e68a9651be100bfd37f30a9607416
=======
>>>>>>> eba84b5f99df32efcb379942275cdda8e709fc87
Test accuracy: 0.93928035982009
"""
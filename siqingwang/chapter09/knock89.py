import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.DataFrame(train)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
model.to(device)

# Create a custom dataset class
class NewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        title = self.dataframe.iloc[index]['TITLE']
        category = self.dataframe.iloc[index]['CATEGORY']
        encoding = self.tokenizer.encode_plus(
            title,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(category, dtype=torch.long)
        }

# Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# Create DataLoaders
train_dataset = NewsDataset(train_df, tokenizer, MAX_LEN)
val_dataset = NewsDataset(val_df, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE)

# Optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss_train = 0
    y_true_train = []
    y_pred_train = []

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        inputs = {
            'input_ids': batch['input_ids'].to(device),
            'attention_mask': batch['attention_mask'].to(device),
            'labels': batch['labels'].to(device)
        }
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss_train += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        y_pred_train.extend(np.argmax(logits, axis=1))
        y_true_train.extend(label_ids)

    avg_train_loss = total_loss_train / len(train_loader)
    train_acc = accuracy_score(y_true_train, y_pred_train)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss}, Training Accuracy: {train_acc}")

    model.eval()
    total_loss_val = 0
    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss_val += loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            y_pred_val.extend(np.argmax(logits, axis=1))
            y_true_val.extend(label_ids)

    avg_val_loss = total_loss_val / len(val_loader)
    val_acc = accuracy_score(y_true_val, y_pred_val)
    print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_acc}")

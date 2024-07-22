import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Parameters
dw = 300  # Dimensionality of word embeddings
dh = 50  # Dimensionality of hidden state
L = 4  # Number of categories
learning_rate = 0.01
epochs = 10
batch_size = 32


class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), dh).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# Example: Function to simulate text to sequences (replace with actual conversion logic)
def text_to_sequences(df, text_column, label_column, mapper):
    sequences = []
    labels = []

    for index, row in df.iterrows():
        words = row[text_column].split()
        sequence = [mapper.get_id(word) for word in words]
        sequences.append(sequence)

        labels.append(row[label_column])

    return sequences, np.array(labels)


# Initialize the mapper
mapper = WordToIDMapper()

# Convert text data to sequences of word IDs and labels
x_train, y_train = text_to_sequences(train, 'TITLE', 'CATEGORY', mapper)

# Split data into training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Create DataLoader for mini-batch training
train_dataset = TextDataset(x_train, y_train)
valid_dataset = TextDataset(x_valid, y_valid)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
vocab_size = max(max(seq) for seq in x_train) + 1
model = RNNModel(vocab_size, dw, dh, L).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    y_preds_train = []
    y_true_train = []

    for sequences, labels in train_loader:
        sequences, labels = sequences.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_preds_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        y_true_train.extend(labels.cpu().numpy())

    train_acc = accuracy_score(y_true_train, y_preds_train)
    print(
        f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}, Training Accuracy: {train_acc}")

    # Validation step
    model.eval()
    total_val_loss = 0
    y_preds_valid = []
    y_true_valid = []

    with torch.no_grad():
        for sequences, labels in valid_loader:
            sequences, labels = sequences.cuda(), labels.cuda()
            outputs = model(sequences)
            val_loss = criterion(outputs, labels)
            total_val_loss += val_loss.item()
            y_preds_valid.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_true_valid.extend(labels.cpu().numpy())

    val_acc = accuracy_score(y_true_valid, y_preds_valid)
    print(f"Validation Loss: {total_val_loss / len(valid_loader)}, Validation Accuracy: {val_acc}")


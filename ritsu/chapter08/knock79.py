# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# import os
# import time
# import matplotlib.pyplot as plt

# class NewsDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X
#         self.y = y

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

# class MultiLayerPerceptron(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
#         super().__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

# def calc_loss_acc(model, criterion, loader, device):
#     model.eval()
#     loss = 0.0
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss += criterion(outputs, labels).item()
#             pred = torch.argmax(outputs, dim=-1)
#             total += labels.size(0)
#             correct += (pred == labels).sum().item()
#     return loss / len(loader), correct / total

# def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device):
#     model.to(device)
#     best_acc = 0.0
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0.0
#         start_time = time.time()

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         train_loss = total_loss / len(train_loader)
#         valid_loss, valid_acc = calc_loss_acc(model, criterion, valid_loader, device)

#         epoch_time = time.time() - start_time
#         print(f'Epoch {epoch+1}/{num_epochs}, '
#               f'Train Loss: {train_loss:.4f}, '
#               f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, '
#               f'Time: {epoch_time:.2f}s')

#         if valid_acc > best_acc:
#             best_acc = valid_acc
#             torch.save(model.state_dict(), 'best_model.pth')

#     return best_acc

# def main():
#     current_dir = os.path.dirname(os.path.abspath(__file__))

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     X_train = torch.load(os.path.join(current_dir, "X_train.pt"))
#     X_valid = torch.load(os.path.join(current_dir, "X_valid.pt"))
#     X_test = torch.load(os.path.join(current_dir, "X_test.pt"))
#     y_train = torch.load(os.path.join(current_dir, "y_train.pt"))
#     y_valid = torch.load(os.path.join(current_dir, "y_valid.pt"))
#     y_test = torch.load(os.path.join(current_dir, "y_test.pt"))

#     dataset_train = NewsDataset(X_train, y_train)
#     dataset_valid = NewsDataset(X_valid, y_valid)
#     dataset_test = NewsDataset(X_test, y_test)

#     batch_size = 64
#     dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
#     dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
#     dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

#     input_size = 300
#     hidden_size = 128
#     output_size = 4
#     model = MultiLayerPerceptron(input_size, hidden_size, output_size)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

#     num_epochs = 30
#     best_acc = train_model(model, criterion, optimizer, dataloader_train, dataloader_valid, num_epochs, device)

#     # Load the best model and evaluate on test set
#     model.load_state_dict(torch.load('best_model.pth'))
#     test_loss, test_acc = calc_loss_acc(model, criterion, dataloader_test, device)
    
#     print(f"\nBest validation accuracy: {best_acc:.4f}")
#     print(f"Test accuracy: {test_acc:.4f}")

# if __name__ == "__main__":
#     main()





import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

class MultiLayerPerceptronNetwork(nn.Module):
    def __init__(self, input_size, mid_size, output_size, mid_layers):
        super().__init__()
        self.mid_layers = mid_layers
        self.fc = nn.Linear(input_size, mid_size)
        self.fc_mid = nn.Linear(mid_size, mid_size)
        self.fc_out = nn.Linear(mid_size, output_size)
        self.bn = nn.BatchNorm1d(mid_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        for _ in range(self.mid_layers):
            x = F.relu(self.bn(self.fc_mid(x)))
        x = self.fc_out(x)
        return x

class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

def calc_acc(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return correct / total

def calc_loss_acc(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss / len(loader), correct / total

def train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device):
    model.to(device)
    
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    
    log_train = []
    log_valid = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        model.train()
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        
        loss_train, acc_train = calc_loss_acc(model, criterion, dataloader_train, device)
        loss_valid, acc_valid = calc_loss_acc(model, criterion, dataloader_valid, device)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, f'checkpoint{epoch + 1}.pt')
        
        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f}, '
              f'loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, '
              f'train_time: {(end_time - start_time):.4f}sec')
    
    return log_train, log_valid

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = torch.load("X_train.pt")
    X_valid = torch.load("X_valid.pt")
    X_test = torch.load("X_test.pt")
    y_train = torch.load("y_train.pt")
    y_valid = torch.load("y_valid.pt")
    y_test = torch.load("y_test.pt")

    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)
    dataset_test = NewsDataset(X_test, y_test)

    model = MultiLayerPerceptronNetwork(300, 200, 4, 1)
    criterion = nn.CrossEntropyLoss()
    # L2正則化を採用するためにweight_decayパラメータを追加
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=1e-5)
    num_epochs = 50
    batch_size = 64

    log_train, log_valid = train_model(dataset_train, dataset_valid, batch_size, model, criterion, optimizer, num_epochs, device)

    # プロットの作成
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(np.array(log_train)[:, 0], label='train')
    ax[0].plot(np.array(log_valid)[:, 0], label='valid')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[1].plot(np.array(log_train)[:, 1], label='train')
    ax[1].plot(np.array(log_valid)[:, 1], label='valid')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('accuracy')
    ax[1].legend()
    plt.savefig("79.png")

    # 最終的な精度の計算
    dataloader_train = DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)
    acc_train = calc_acc(model, dataloader_train, device)
    acc_test = calc_acc(model, dataloader_test, device)
    print(f"train_acc : {acc_train}")
    print(f"test_acc : {acc_test}")

if __name__ == "__main__":
    main()

"""
一番上の基本コードにL2正則化を加えた場合

train_acc : 0.9990629685157422
test_acc : 0.9100449775112444
"""
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from knock71 import SingleLayerPerceptronNetwork

class NewsDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]

def calc_acc(model, loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return correct / total

def calc_loss_acc(model, criterion, loader):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs, dim=-1)
            total += len(inputs)
            correct += (pred == labels).sum().item()
    return loss / len(loader), correct / total

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    X_train = torch.load(os.path.join(current_dir, "X_train.pt"))
    X_valid = torch.load(os.path.join(current_dir, "X_valid.pt"))
    X_test = torch.load(os.path.join(current_dir, "X_test.pt"))
    y_train = torch.load(os.path.join(current_dir, "y_train.pt"))
    y_valid = torch.load(os.path.join(current_dir, "y_valid.pt"))
    y_test = torch.load(os.path.join(current_dir, "y_test.pt"))

    dataset_train = NewsDataset(X_train, y_train)
    dataset_valid = NewsDataset(X_valid, y_valid)
    dataset_test = NewsDataset(X_test, y_test)

    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
    dataloader_valid = DataLoader(dataset_valid, batch_size=len(dataset_valid), shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)

    model = SingleLayerPerceptronNetwork(X_train.shape[1], 4)
    criterion = nn.CrossEntropyLoss()

    # 1つ目の事例での損失と勾配
    l_1 = criterion(model(X_train[:1]), y_train[:1])
    model.zero_grad()
    l_1.backward()
    print(f'1つ目の事例の損失: {l_1:.4f}')
    print(f'1つ目の事例の勾配:\n{model.fc.weight.grad}')

    # 最初の4つの事例での損失と勾配
    l = criterion(model(X_train[:4]), y_train[:4])
    model.zero_grad()
    l.backward()
    print(f'\n最初の4つの事例の損失: {l:.4f}')
    print(f'最初の4つの事例の勾配:\n{model.fc.weight.grad}')

    # 検証データでの損失と正解率
    valid_loss, valid_acc = calc_loss_acc(model, criterion, dataloader_valid)
    print(f'\n検証データでの損失: {valid_loss:.4f}')
    print(f'検証データでの正解率: {valid_acc:.4f}')

if __name__ == "__main__":
    main()

"""
1つ目の事例の損失: 3.2087
1つ目の事例の勾配:
tensor([[ 0.0126, -0.0144, -0.0246,  ..., -0.0244,  0.0210,  0.0105],
        [ 0.0131, -0.0150, -0.0255,  ..., -0.0254,  0.0218,  0.0109],
        [-0.0274,  0.0315,  0.0536,  ...,  0.0533, -0.0458, -0.0230],
        [ 0.0018, -0.0021, -0.0035,  ..., -0.0035,  0.0030,  0.0015]])

最初の4つの事例の損失: 3.2263
最初の4つの事例の勾配:
tensor([[-0.0155, -0.0363,  0.0286,  ..., -0.0343, -0.0059,  0.0322],
        [ 0.0305,  0.0172, -0.0301,  ...,  0.0197, -0.0054, -0.0283],
        [-0.0086,  0.0150,  0.0080,  ...,  0.0226,  0.0021, -0.0057],
        [-0.0064,  0.0041, -0.0065,  ..., -0.0080,  0.0092,  0.0019]])

検証データでの損失: 2.5193
検証データでの正解率: 0.1559
"""
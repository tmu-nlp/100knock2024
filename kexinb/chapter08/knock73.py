# task 73. 確率的勾配降下法による学習

from knock72 import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


dataset_train = NewsDataset(X_train, y_train)
dataset_valid = NewsDataset(X_valid, y_valid)
dataset_test = NewsDataset(X_test, y_test)

dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)
dataloader_valid = DataLoader(
    dataset_valid, batch_size=len(dataset_valid), shuffle=False)
dataloader_test = DataLoader(
    dataset_test, batch_size=len(dataset_test), shuffle=False)



model = SingleLayerPerceptronNetwork(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    loss_train = 0.0
    for i, (inputs, labels) in enumerate(dataloader_train):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
    loss_train = loss_train / i
    model.eval()
    with torch.no_grad():
        inputs, labels = next(iter(dataloader_valid))
        outputs = model(inputs)
        loss_valid = criterion(outputs, labels)

    # print(
    #     f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, loss_valid: {loss_valid:.4f}')

'''
epoch: 1, loss_train: 0.4537, loss_valid: 0.3773
epoch: 2, loss_train: 0.3062, loss_valid: 0.3451
epoch: 3, loss_train: 0.2774, loss_valid: 0.3269
epoch: 4, loss_train: 0.2623, loss_valid: 0.3277
epoch: 5, loss_train: 0.2524, loss_valid: 0.3190
epoch: 6, loss_train: 0.2454, loss_valid: 0.3210
epoch: 7, loss_train: 0.2398, loss_valid: 0.3188
epoch: 8, loss_train: 0.2358, loss_valid: 0.3160
epoch: 9, loss_train: 0.2324, loss_valid: 0.3192
epoch: 10, loss_train: 0.2297, loss_valid: 0.3182
'''
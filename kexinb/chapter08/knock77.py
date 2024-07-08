# task77. ミニバッチ化

from knock72 import *
from torch.utils.data import DataLoader
import time


def train_model(dataset_train, dataset_valid, batch_size, model, criterion, num_epochs):
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    log_train = []
    log_valid = []

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        loss_train = 0.0
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        end_time = time.time()

        loss_train, acc_train = calc_loss_acc(
            model, criterion, dataloader_train)
        loss_valid, acc_valid = calc_loss_acc(
            model, criterion, dataloader_valid)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        # torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(
        # ), 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoint{epoch + 1}.pt')

        print(f'epoch: {epoch + 1}, loss_train: {loss_train:.4f}, accuracy_train: {acc_train:.4f},\
              loss_valid: {loss_valid:.4f}, accuracy_valid: {acc_valid:.4f}, \
              train_time: {(end_time - start_time):.4f}sec')


model = SingleLayerPerceptronNetwork(300, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
num_epochs = 1

for batch_size in [2 ** i for i in range(12)]:
    print(f"batch_size : {batch_size}")
    train_model(dataset_train, dataset_valid,
                batch_size, model, criterion, num_epochs)
    
'''
batch_size : 1
    epoch: 1, loss_train: 0.3156, accuracy_train: 0.8908,
    loss_valid: 0.3809, accuracy_valid: 0.8725,
    train_time: 1.4668sec
batch_size : 2
    epoch: 1,   loss_train: 0.2917, accuracy_train: 0.8965,
    loss_valid: 0.3531, accuracy_valid: 0.8807,
    train_time: 1.4872sec
batch_size : 4
    epoch: 1, loss_train: 0.2824, accuracy_train: 0.9025,
    loss_valid: 0.3463, accuracy_valid: 0.8800,
    train_time: 0.7602sec
batch_size : 8
    epoch: 1, loss_train: 0.2783, accuracy_train: 0.9039,
    loss_valid: 0.3441, accuracy_valid: 0.8822,
    train_time: 0.4271sec
batch_size : 16
    epoch: 1, loss_train: 0.2767, accuracy_train: 0.9033,
    loss_valid: 0.3435, accuracy_valid: 0.8837,
    train_time: 0.2753sec
batch_size : 32
    epoch: 1, loss_train: 0.2755, accuracy_train: 0.9040,
    loss_valid: 0.3425, accuracy_valid: 0.8837,
    train_time: 0.1435sec
batch_size : 64
    epoch: 1, loss_train: 0.2753, accuracy_train: 0.9041,
    loss_valid: 0.3419, accuracy_valid: 0.8845,
    train_time: 0.1668sec
batch_size : 128
    epoch: 1, loss_train: 0.2749, accuracy_train: 0.9040,
    loss_valid: 0.3418, accuracy_valid: 0.8845,
    train_time: 0.1565sec
batch_size : 256
    epoch: 1, loss_train: 0.2751, accuracy_train: 0.9040,
    loss_valid: 0.3417, accuracy_valid: 0.8845,
    train_time: 0.1295sec
batch_size : 512
    epoch: 1, loss_train: 0.2754, accuracy_train: 0.9040,
    loss_valid: 0.3417, accuracy_valid: 0.8845,
    train_time: 0.1240sec
batch_size : 1024
    epoch: 1, loss_train: 0.2723, accuracy_train: 0.9040,
    loss_valid: 0.3416, accuracy_valid: 0.8845,
    train_time: 0.0543sec
batch_size : 2048
    epoch: 1, loss_train: 0.2744, accuracy_train: 0.9041,
    loss_valid: 0.3416, accuracy_valid: 0.8845,
    train_time: 0.0609sec
''' 
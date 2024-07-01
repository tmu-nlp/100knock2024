# task71. 単層ニューラルネットワークによる予測

from torch import nn
import torch

class SingleLayerPerceptronNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc(x)
        return x


X_train = torch.load("output/ch8/X_train.pt")
model = SingleLayerPerceptronNetwork(300, 4)
y_hat_1 = torch.softmax(model(X_train[:1]), dim=-1)
Y_hat = torch.softmax(model.forward(X_train[:4]), dim=-1)

# print(y_hat_1)
# print(Y_hat)

'''
tensor([[0.0228, 0.7676, 0.1454, 0.0643]], grad_fn=<SoftmaxBackward0>)
tensor([[0.0228, 0.7676, 0.1454, 0.0643],
        [0.1204, 0.5921, 0.2185, 0.0690],
        [0.1431, 0.6751, 0.1615, 0.0203],
        [0.2149, 0.6398, 0.0748, 0.0706]], grad_fn=<SoftmaxBackward0>)
'''
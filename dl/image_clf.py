import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        # in_channels=3, out_channels=4, kernel_size=3x3
        self.cnn1 = nn.Conv2d(1, 4, 3)
        self.cnn2 = nn.Conv2d(4, 8, 3)
        self.cnn3 = nn.Conv2d(8, 16, 5)
        self.cnn4 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2)
        self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(32, 32)
        self.out = nn.Linear(32, n_classes)

    def forward(self, x):
        act = self.activation
        x = act(self.cnn1(x))
        x = act(self.cnn2(x))
        x = self.pool(x)
        x = act(self.cnn3(x))
        x = act(self.cnn4(x))
        x = self.pool(x)
        x = x.mean(dim=-1).mean(dim=-1)
        x = act(self.fc(x))
        return self.out(x)


class Trainer:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size

    def train(self, data, n_epochs=10):
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())

        for i in range(n_epochs):
            for i_batch, batch in enumerate(loader):
                y_pred = self.model(batch[0])
                # no softmax
                loss = F.cross_entropy(y_pred, batch[1])

                # set grad to zero
                optimizer.zero_grad()
                # loss bp
                loss.backward()
                # optimize params
                optimizer.step()

            print('epoch:', i, 'loss:', loss.mean().item())

    def evaluate(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

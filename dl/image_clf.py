import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, dropout_p=0.5):
        super().__init__()
        self.n_classes = n_classes
        act = self.act = nn.ReLU()
        dropout = self.dropout = nn.Dropout(dropout_p)
        self.feature = nn.Sequential(
            # in_channels=1, out_channels=4, kernel_size=3x3
            dropout,
            nn.Conv2d(1, 4, 3),
            act,
            dropout,
            nn.Conv2d(4, 8, 3),
            act,
            nn.MaxPool2d(2),
            dropout,
            nn.Conv2d(8, 16, 3),
            act,
            dropout,
            nn.Conv2d(16, 32, 3),
            act,
            nn.MaxPool2d(2),
            dropout,
            nn.Conv2d(32, 64, 3),
            act,
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64, 32)
        self.out = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.feature(x)
        x = x.mean(dim=-1).mean(dim=-1)
        x = self.act(self.fc(self.dropout(x)))
        return self.out(self.dropout(x))


class Trainer:
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size

    def train(self, data, n_epochs=10):
        # train mode
        self.model.train()
        # TODO: add validation dataset
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters())

        for i in range(n_epochs):
            # tqdm for progress bar.
            # Set file to avoid conflict with print in Jupyter.
            for i_batch, batch in tqdm(enumerate(loader), total=len(loader),
                                       file=sys.stdout):
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

    def evaluate(self, data):
        # evaluation mode
        self.model.eval()
        loader = DataLoader(data, batch_size=2 * self.batch_size)

        # predict result without computing grad
        with torch.no_grad():
            predictions = []
            groundtruth = []
            for i_batch, batch in tqdm(enumerate(loader), total=len(loader),
                                       file=sys.stdout):
                y_pred = self.model(batch[0])
                y_pred = y_pred.max(dim=1)[1]
                y_true = batch[1]
                predictions.append(y_pred)
                groundtruth.append(y_true)
            y_pred = torch.cat(predictions)
            y_true = torch.cat(groundtruth)
        return precision_recall_fscore_support(
            y_true, y_pred, average='weighted')


import numpy as np
import torch.nn as nn
import torch
from sklearn.base import BaseEstimator

from torchmtlr import MTLR, mtlr_neg_log_likelihood
from torchmtlr.utils import encode_survival


class MTLRWrapper(BaseEstimator):
    def __init__(self, input_dim, time_bins, n_hidden1=64, n_hidden2=32, dropout1=0.2, dropout2=0.2, activation='relu',
                 n_epochs=100, lr=0.001, C1=1.0):
        self.input_dim = input_dim
        self.time_bins = time_bins
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.activation = activation
        self.n_epochs = n_epochs
        self.lr = lr
        self.C1 = C1
        self.model = None

    def fit(self, X, y):
        # y est un numpy structured array avec les champs 'event' et 'time'
        # On force la copie pour éviter le bug de strides
        y_event = torch.tensor(np.copy(y['OS_STATUS'] if 'OS_STATUS' in y.dtype.names else y['event']),
                               dtype=torch.float32)
        y_time = torch.tensor(np.copy(y['OS_YEARS'] if 'OS_YEARS' in y.dtype.names else y['time']), dtype=torch.float32)
        X_tensor = torch.tensor(np.copy(X.values), dtype=torch.float32)
        target = encode_survival(y_time, y_event, self.time_bins)
        act_fn = nn.ReLU() if self.activation == 'relu' else nn.ELU()
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.n_hidden1),
            act_fn,
            nn.Dropout(self.dropout1),
            nn.Linear(self.n_hidden1, self.n_hidden2),
            act_fn,
            nn.Dropout(self.dropout2),
            MTLR(self.n_hidden2, len(self.time_bins))
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.model(X_tensor)
            loss = mtlr_neg_log_likelihood(logits, target, self.model[-1], C1=self.C1, average=True)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        import torch
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            risk_scores = torch.logsumexp(logits, dim=1).numpy()
        return risk_scores

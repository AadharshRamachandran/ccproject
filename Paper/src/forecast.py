from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class BiLSTM(nn.Module):
    def __init__(self, hidden=32):
        super().__init__(); self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True); self.out = nn.Linear(hidden * 2, 1)
    def forward(self, x): return self.out(self.lstm(x)[0][:, -1, :])

class WorkloadForecaster:
    """Paper Bi-LSTM: 10 historic points predict the next normalized rate."""
    def __init__(self, lookback=10, hidden=32):
        self.lookback, self.model, self.minimum, self.maximum = lookback, BiLSTM(hidden), 0., 1.
    def fit(self, values, epochs=80, learning_rate=.005, batch_size=512):
        values = np.asarray(values, dtype=np.float32); self.minimum, self.maximum = float(values.min()), float(values.max())
        span = max(self.maximum - self.minimum, 1e-9); norm = (values-self.minimum)/span
        x = np.array([norm[i-self.lookback:i] for i in range(self.lookback, len(norm))]); y = norm[self.lookback:]
        if not len(x): raise ValueError('Need more data than lookback')
        X = torch.tensor(x).unsqueeze(-1); Y = torch.tensor(y).unsqueeze(-1)
        batches = DataLoader(TensorDataset(X,Y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate); loss = nn.MSELoss(); self.model.train()
        # Do not run every window through one autograd graph: World Cup '98 has
        # over 100k minute points, which otherwise needs multiple GB of RAM.
        for _ in range(epochs):
            for batch_x,batch_y in batches:
                optimizer.zero_grad(); value=loss(self.model(batch_x),batch_y); value.backward(); optimizer.step()
    def predict(self, history):
        if len(history) < self.lookback: return float(history[-1]) if history else 0.
        span = max(self.maximum-self.minimum, 1e-9); x = (np.asarray(history[-self.lookback:], dtype=np.float32)-self.minimum)/span
        self.model.eval()
        with torch.no_grad(): normalized = float(self.model(torch.tensor(x).view(1, self.lookback, 1)).item())
        return max(0., normalized*span+self.minimum)
    def save(self, path): torch.save({'state':self.model.state_dict(),'minimum':self.minimum,'maximum':self.maximum,'lookback':self.lookback}, path)
    @classmethod
    def load(cls, path):
        saved=torch.load(path, map_location='cpu', weights_only=True); obj=cls(saved['lookback']); obj.model.load_state_dict(saved['state']); obj.minimum=saved['minimum']; obj.maximum=saved['maximum']; return obj

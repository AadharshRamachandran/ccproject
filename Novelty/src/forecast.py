"""Workload forecasters: paper Bi-LSTM baseline plus novelty quantile-BiLSTM.

All forecasters live here so that downstream code (controller, ablation, training
scripts) can import from a single authoritative location:

    from Novelty.src.forecast import (
        BiLSTM, WorkloadForecaster,         # paper baseline
        QuantileBiLSTMForecaster,           # novelty probabilistic
        FORECASTERS,                        # registry used by controller
    )
"""
from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from Paper.src.forecast import BiLSTM, WorkloadForecaster


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _windows(values, lookback, stride=1):
    """Create supervised windows, optionally retaining every ``stride``-th one.

    World Cup minute observations are strongly autocorrelated.  Sampling
    overlapping windows during *training* therefore avoids thousands of nearly
    identical gradient updates without changing the chronological split.
    """
    values = np.asarray(values, dtype=np.float32)
    if stride < 1:
        raise ValueError('stride must be at least 1')
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback + 1)[::stride]
    return windows[:, :-1].copy(), windows[:, -1].copy()


def _report_epoch(label, epoch, epochs, loss, progress_every):
    if progress_every and (epoch == 1 or epoch % progress_every == 0 or epoch == epochs):
        print(f'[{label}] epoch {epoch}/{epochs} loss={loss:.6f}', flush=True)


# ---------------------------------------------------------------------------
# The deterministic baseline is imported from Paper/src/forecast.py.

# Novelty: Probabilistic Quantile Bi-LSTM
# ---------------------------------------------------------------------------

class _QuantileBiLSTM(nn.Module):
    def __init__(self, hidden=32, dropout=0.15):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden * 2, 3)  # p10, p50, p90

    def forward(self, x):
        return self.out(self.dropout(self.lstm(x)[0][:, -1, :]))


class QuantileBiLSTMForecaster:
    """P10/P50/P90 Bi-LSTM.  The p90 safety margin drives uncertainty provisioning."""

    quantiles = torch.tensor([0.10, 0.50, 0.90])

    def __init__(self, lookback=10, hidden=32, dropout=0.15):
        self.lookback = lookback
        self.hidden = hidden
        self.dropout = dropout
        self.model = _QuantileBiLSTM(hidden, dropout)
        self.minimum, self.maximum = 0., 1.

    def fit(self, values, epochs=80, learning_rate=.005, batch_size=512,
            window_stride=1, progress_every=0, label='quantile_bilstm', seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        values = np.asarray(values, dtype=np.float32)
        self.minimum, self.maximum = float(values.min()), float(values.max())
        scale = max(self.maximum - self.minimum, 1e-9)
        x, y = _windows((values - self.minimum) / scale, self.lookback, window_stride)
        if not len(x):
            raise ValueError('Need more data than lookback')
        X = torch.tensor(x).unsqueeze(-1)
        # Learn quantile residuals around the last observed level.  The
        # skip connection gives each quantile a persistence baseline and lets
        # the network focus on the next-step change.
        Y = torch.tensor(y - x[:, -1]).unsqueeze(-1)
        batches = DataLoader(TensorDataset(X, Y), batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        q = self.quantiles.view(1, -1)
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            batches_seen = 0
            for batch_x, batch_y in batches:
               optimizer.zero_grad()
               error = batch_y - self.model(batch_x)
               loss = torch.maximum(q * error, (q - 1) * error).mean()
               loss.backward()
               optimizer.step()
               epoch_loss += float(loss.detach())
               batches_seen += 1
            _report_epoch(label, epoch, epochs, epoch_loss / max(batches_seen, 1), progress_every)

    def predict_interval(self, history):
        if len(history) < self.lookback:
            point = float(history[-1]) if history else 0.
            return point, point, point
        scale = max(self.maximum - self.minimum, 1e-9)
        x = (np.asarray(history[-self.lookback:], dtype=np.float32) - self.minimum) / scale
        self.model.eval()
        with torch.no_grad():
            residuals = self.model(torch.tensor(x).view(1, self.lookback, 1))[0].numpy()
        values = residuals + x[-1]
        values = np.sort(values) * scale + self.minimum
        return tuple(float(max(0., v)) for v in values)

    def predict_intervals_many(self, history, values, batch_size=4096):
        """Batch teacher-forced one-step quantile predictions."""
        values = np.asarray(values, dtype=np.float32)
        context = np.concatenate((np.asarray(history, dtype=np.float32)[-self.lookback:], values))
        windows = np.lib.stride_tricks.sliding_window_view(context, self.lookback + 1)[:len(values), :-1]
        scale = max(self.maximum - self.minimum, 1e-9)
        windows = (windows - self.minimum) / scale
        outputs=[]; self.model.eval()
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                x=torch.as_tensor(windows[start:start+batch_size]).unsqueeze(-1)
                residuals=self.model(x).cpu().numpy()
                outputs.extend((residuals + windows[start:start+batch_size, -1, None]).tolist())
        intervals=np.sort(np.asarray(outputs),axis=1)*scale+self.minimum
        return np.maximum(0., intervals)

    def save(self, path):
        torch.save({
            'state': self.model.state_dict(),
            'minimum': self.minimum,
            'maximum': self.maximum,
            'lookback': self.lookback,
            'hidden': self.hidden,
            'dropout': self.dropout,
            'target_mode': 'residual_quantiles_v1',
        }, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('target_mode') != 'residual_quantiles_v1':
            raise ValueError('This quantile checkpoint predates residual quantile training; retrain it with scripts/train_novelty.py.')
        obj = cls(saved['lookback'], saved['hidden'], saved['dropout'])
        obj.model.load_state_dict(saved['state'])
        obj.minimum = saved['minimum']
        obj.maximum = saved['maximum']
        return obj


# ---------------------------------------------------------------------------
# Registry (used by controller/main.py)
# ---------------------------------------------------------------------------

FORECASTERS = {
    'bilstm': WorkloadForecaster,
    'quantile_bilstm': QuantileBiLSTMForecaster,
}

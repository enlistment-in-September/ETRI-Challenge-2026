import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.utils import clip_probs


class SleepSequenceDataset(Dataset):
    def __init__(self, sample_df, subject_daily_dict, subject2idx, feature_cols,
                 lookback=14, targets=None, is_test=False):
        self.df = sample_df.reset_index(drop=True).copy()
        self.subject_daily_dict = subject_daily_dict
        self.subject2idx = subject2idx
        self.feature_cols = feature_cols
        self.lookback = lookback
        self.targets = targets
        self.is_test = is_test
        self.df["date_dt"] = pd.to_datetime(self.df["lifelog_date"])

    def __len__(self):
        return len(self.df)

    def _make_seq(self, subject_id, current_date):
        info = self.subject_daily_dict[subject_id]
        dates = info["date_dt"]
        feat = info["feat"]
        idxs = [i for i, d in enumerate(dates) if d <= current_date][-self.lookback:]
        seq = np.zeros((self.lookback, feat.shape[1]), dtype=np.float32)
        mask = np.zeros((self.lookback,), dtype=np.float32)
        if idxs:
            use_feat = np.nan_to_num(feat[idxs], nan=0.0, posinf=0.0, neginf=0.0)
            seq[-len(idxs):] = use_feat
            mask[-len(idxs):] = 1.0
        return np.nan_to_num(seq), np.nan_to_num(mask)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sid = row["subject_id"]
        seq, mask = self._make_seq(sid, row["date_dt"])
        item = {
            "seq": torch.tensor(seq, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "static": torch.tensor(seq[-1].copy(), dtype=torch.float32),
            "subject_id": torch.tensor(self.subject2idx[sid], dtype=torch.long),
        }
        if not self.is_test:
            y = np.nan_to_num(row[self.targets].values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
            item["target"] = torch.tensor(y, dtype=torch.float32)
        return item


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, mask):
        score = self.attn(x).squeeze(-1)
        score = score.masked_fill(mask == 0, -1e9)
        weight = torch.softmax(score, dim=1)
        pooled = torch.nan_to_num(torch.bmm(weight.unsqueeze(1), x).squeeze(1))
        return pooled, weight


class MultiTaskBiLSTM(nn.Module):
    def __init__(self, input_dim, n_subjects, hidden_dim=96, num_layers=2, dropout=0.25, out_dim=7):
        super().__init__()
        self.subject_emb = nn.Embedding(n_subjects, 12)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn_pool = AttentionPooling(hidden_dim * 2)
        self.static_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        fusion_dim = hidden_dim * 2 + hidden_dim * 2 + hidden_dim + 12
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, seq, mask, static, subject_id):
        seq = torch.nan_to_num(seq)
        static = torch.nan_to_num(static)
        x = torch.nan_to_num(self.input_proj(seq))
        lstm_out = torch.nan_to_num(self.lstm(x)[0])
        attn_vec, _ = self.attn_pool(lstm_out, mask)
        lengths = mask.sum(dim=1).long().clamp(min=1)
        last_vec = lstm_out[torch.arange(seq.size(0), device=seq.device), lengths - 1]
        fused = torch.nan_to_num(torch.cat([attn_vec, last_vec, self.static_mlp(static), self.subject_emb(subject_id)], dim=1))
        return torch.nan_to_num(self.head(fused))


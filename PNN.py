import torch
import torch.nn as nn


class PNN1(nn.Module):
    def __init__(self, num_inputs, embed_size, hidden_dim, keep_prob):
        super().__init__()
        self.num_inputs = num_inputs
        self.embed_size = embed_size
        self.num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        self.row = []
        self.col = []
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                self.row.append(i)
                self.col.append(j)
        self.fc1 = nn.Linear(num_inputs * embed_size + self.num_pairs, hidden_dim)
        self.dropout = nn.Dropout(p=1 - keep_prob)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, inputs):
        xw = torch.cat(inputs, dim=1)
        xw3d = xw.view(-1, self.num_inputs, self.embed_size)
        p = xw3d[:, self.row, :]
        q = xw3d[:, self.col, :]
        ip = torch.sum(p * q, dim=-1)
        l = torch.cat([xw, ip], dim=1)
        h = torch.relu(self.fc1(l))
        h = self.dropout(h)
        pred = self.fc2(h).squeeze(-1)
        return h, pred


def pnn1(inputs, embed_size, hidden_dim, keep_prob):
    model = PNN1(num_inputs=len(inputs), embed_size=embed_size, hidden_dim=hidden_dim, keep_prob=keep_prob)
    return model(inputs)

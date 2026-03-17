import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(np.array(ns)).to(device),
            torch.FloatTensor(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)
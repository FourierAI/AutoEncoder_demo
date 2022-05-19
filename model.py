import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.structure = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, out_dim),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.structure(x)
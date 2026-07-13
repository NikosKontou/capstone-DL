import torch
import torch.nn as nn
import config as cfg

class ResBlock(nn.Module):
    def __init__(self, dim, dropout, use_batchnorm):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.relu(self.bn(self.lin(x))))

class SpinMultiTaskNet(nn.Module):
    def __init__(self, n_features: int, hidden_dims=None, dropout: float = None,
                 use_batchnorm: bool = None):
        super().__init__()
        hidden_dims = hidden_dims or cfg.HIDDEN_DIMS
        dropout = cfg.DROPOUT if dropout is None else dropout
        use_batchnorm = cfg.USE_BATCHNORM if use_batchnorm is None else use_batchnorm

        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(ResBlock(h, dropout, use_batchnorm))
            in_dim = h

        self.trunk = nn.Sequential(*layers)

        self.head_spins_left = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()
        )
        self.head_next_bet = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus()
        )

    def forward(self, x):
        z = self.trunk(x)
        return self.head_spins_left(z), self.head_next_bet(z)

def build_model(n_features: int) -> SpinMultiTaskNet:
    return SpinMultiTaskNet(n_features=n_features)

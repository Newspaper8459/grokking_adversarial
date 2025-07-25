import torch
import torch.nn as nn


class MLP(nn.Module):
  def __init__(self, hidden_dim: int, hidden_layers: int):
    super().__init__()
    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(784, hidden_dim),
      nn.ReLU(),
      *[nn.Linear(hidden_dim, hidden_dim) if i%2 == 0 else nn.ReLU() for i in range(2*hidden_layers)],
      nn.Linear(hidden_dim, 10)
    )

  def forward(self, x: torch.Tensor):
    return self.fc(x)

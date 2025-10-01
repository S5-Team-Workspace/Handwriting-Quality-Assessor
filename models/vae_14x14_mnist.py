"""
VAE model extracted from VAE_Digit_Recognition.ipynb (14x14 MNIST, binarized)
Architecture: 196 -> 128 -> 8 (mu, std) -> 128 -> 196 with tanh/sigmoid activations.
Includes helpers to save/load the model state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE14x14(nn.Module):
    def __init__(self, input_dim: int = 196, hidden_dim: int = 128, latent_dim: int = 8):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # std (not logvar in original)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.fc1(x))
        return self.fc21(h), self.fc22(h)  # mu, std

    def sampling(self, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # Reparameterization trick with two samples averaged (as in notebook)
        eps1 = torch.randn_like(std)
        eps2 = torch.randn_like(std)
        return 0.5 * ((eps1 * std + mu) + (eps2 * std + mu))

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x is expected as (B, 196)
        mu, std = self.encoder(x)
        z = self.sampling(mu, std)
        return self.decoder(z), mu, std


def save_state_dict(model: VAE14x14, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_state_dict(path: str, device: str | torch.device | None = None) -> VAE14x14:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(path, map_location=device)
    model = VAE14x14()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

"""
src/dl/model.py
===============
PyTorch MLP for cybersecurity incident risk prediction.

Architecture
------------
    Input: 7 features (same as RF pipeline)
    Layer 1: Linear(7 → 64) + BatchNorm + ReLU + Dropout(0.3)
    Layer 2: Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.2)
    Output:  Linear(32 → 1) + Sigmoid  →  P(high_risk) ∈ [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CyberMLP(nn.Module):
    """
    3-layer Feed-Forward Neural Network for binary risk classification.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 7).
    hidden1 : int
        Units in the first hidden layer (default 64).
    hidden2 : int
        Units in the second hidden layer (default 32).
    dropout1 : float
        Dropout probability after layer 1 (default 0.3).
    dropout2 : float
        Dropout probability after layer 2 (default 0.2).
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden1: int = 64,
        hidden2: int = 32,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            # --- Block 1 ---
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            # --- Block 2 ---
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            # --- Output ---
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return risk probability for a batch of feature vectors."""
        return self.net(x).squeeze(1)

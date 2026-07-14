from __future__ import annotations

import torch
from torch import nn

from src.utils import INPUT_DIM_FLAT, MOVE_PLANES

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Pre-activation residual block with two 3×3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Main model – CNN + Dueling DQN
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """Convolutional Dueling-DQN for chess.

    Input : (batch, 18, 8, 8) board tensor produced by ``encode_board``.
    Output: (batch, MOVE_PLANES) Q-values for every possible action index.

    Architecture
    ------------
    1. Input convolution  18 → ``channels``
    2. ``num_res_blocks`` residual blocks
    3. Two heads (Dueling):
       • **Advantage head** – per-action advantage  A(s, a)
       • **Value head**     – scalar state value     V(s)
       Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
    """

    def __init__(
        self,
        in_channels: int = 18,
        num_res_blocks: int = 6,
        channels: int = 128,
        output_dim: int = MOVE_PLANES,
    ):
        super().__init__()
        self.in_channels = in_channels

        # --- shared trunk ---
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # --- advantage head (per-action) ---
        self.advantage_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        # --- value head (scalar) ---
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 18, 8, 8)
        h = self.input_conv(x)
        h = self.res_blocks(h)
        advantage = self.advantage_head(h)           # (batch, MOVE_PLANES)
        value = self.value_head(h)                    # (batch, 1)
        # Dueling aggregation
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q


# ---------------------------------------------------------------------------
# Legacy flat MLP kept for loading old checkpoints
# ---------------------------------------------------------------------------

class DQNFlat(nn.Module):
    """Original 3-layer MLP (for backward-compatible checkpoint loading)."""

    def __init__(self, input_dim: int = INPUT_DIM_FLAT, output_dim: int = MOVE_PLANES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

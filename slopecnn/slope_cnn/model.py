"""
slope_cnn/model.py
==================
The two CNNs compared in the paper. Both take the same input tensor
(B, C, 42, 243) and return the predicted FoS, shape (B,).

SlopeCNN (proposed)
-------------------
  Stem       : Conv(C->12, 1x7) + BN + ReLU      (reads along the near-horizontal layers)
               Conv(12->16, 3x3) + BN + ReLU
  MaxPool(2, 6)
  ResBlock-1 : [Conv(16->16, 3x3) + BN + ReLU, Conv(16->16, 3x3) + BN] x SE gate + identity skip, ReLU
  MaxPool(3, 4)
  ResBlock-2 : [Conv(16->24, 3x3) + BN + ReLU, Conv(24->24, 3x3) + BN] x SE gate + 1x1 projection skip, ReLU
  Dropout2d(0.2), Flatten, concat fill_ratio
  Head       : Linear(->64) + ReLU + Dropout(0.4) + Linear(64->1)

  fill_ratio = fraction of soil cells in the image (channel 0 > 0). It is now
  computed INSIDE the model from the input, so training and evaluation use the
  same value. (In an earlier version it was real during training and 0 during
  evaluation.)

  Parameters with 6 channels on 42x243: 124,259 (16,546 convolutional + 107,713 head).

Soranzo-type baseline (Soranzo et al. 2023, Acta Geotechnica 18:3391-3402)
---------------------------------------------------------------------------
  [Conv(3x3, 'same') + LeakyReLU(0.1) + MaxPool(2)] x 3 with 32, 64, 128 filters
  Flatten, Dense(128) + LeakyReLU(0.1), Dense(1)

  Same layer list as the authors' published Keras file
  (11440_2022_1783_MOESM9_ESM.h5). Note: that file declares its input as
  [None, 3, 100, 100] with channels_last, so Keras treats it as a 3 x 100 image
  with 100 channels; that is where the paper's 334,433 parameters come from.
  Built as the paper describes it (C channels on an H x W image), the network
  has ~2.55 M parameters on 42x243 (2,551,969 with 6 channels).

  Here it runs on the native 42x243 grid (no resampling to 100x100), with the
  same input scaling as SlopeCNN.
"""

from typing import Tuple

import torch
import torch.nn as nn


def _se_block(ch: int, ratio: int = 4) -> nn.Sequential:
    """Squeeze-and-Excitation: global context -> per-channel [0, 1] gate."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(ch, ch // ratio), nn.ReLU(),
        nn.Linear(ch // ratio, ch), nn.Sigmoid(),
    )


class _ResBlock(nn.Module):
    """Two 3x3 convs + BN, SE-gated, identity skip (same channel count)."""

    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.se = _se_block(ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        return self.relu(h * self.se(h)[:, :, None, None] + x)


def _flat_size(features: nn.Module, in_ch: int, hw: Tuple[int, int]) -> int:
    was_training = features.training
    features.eval()
    with torch.no_grad():
        n = int(features(torch.zeros(1, in_ch, *hw)).numel())
    features.train(was_training)
    return n


class SlopeCNN(nn.Module):
    """Proposed model. forward(x) -> (B,) FoS."""

    def __init__(self, in_ch: int = 6, input_hw: Tuple[int, int] = (42, 243),
                 dropout_rate: float = 0.4, use_fill_ratio: bool = True):
        super().__init__()
        self.use_fill_ratio = use_fill_ratio
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 12, (1, 7), padding=(0, 3), bias=False), nn.BatchNorm2d(12), nn.ReLU(),
            nn.Conv2d(12, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d((2, 6))
        self.res1 = _ResBlock(16)
        self.pool2 = nn.MaxPool2d((3, 4))
        self.res2_conv = nn.Sequential(
            nn.Conv2d(16, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 24, 3, padding=1, bias=False), nn.BatchNorm2d(24),
        )
        self.res2_se = _se_block(24)
        self.res2_proj = nn.Conv2d(16, 24, 1, bias=False)
        self.res2_relu = nn.ReLU()
        self.drop2d = nn.Dropout2d(0.2)
        n_flat = _flat_size(nn.Sequential(self.stem, self.pool1, self.res1, self.pool2,
                                          self.res2_proj), in_ch, input_hw)
        self.head = nn.Sequential(
            nn.Linear(n_flat + (1 if use_fill_ratio else 0), 64), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool2(self.res1(self.pool1(self.stem(x))))
        h2 = self.res2_conv(h)
        h = self.res2_relu(h2 * self.res2_se(h2)[:, :, None, None] + self.res2_proj(h))
        h = torch.flatten(self.drop2d(h), 1)
        if self.use_fill_ratio:
            fill = (x[:, 0] > 0).float().mean(dim=(1, 2))
            h = torch.cat([h, fill[:, None]], dim=1)
        return self.head(h).squeeze(-1)


class SoranzoCNN(nn.Module):
    """Soranzo et al. (2023) layer list on the native grid. forward(x) -> (B,) FoS."""

    def __init__(self, in_ch: int = 3, input_hw: Tuple[int, int] = (42, 243)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
        )
        n_flat = _flat_size(self.features, in_ch, input_hw)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flat, 128), nn.LeakyReLU(0.1), nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


MODELS = {"slopecnn": SlopeCNN, "soranzo": SoranzoCNN}


def build_model(name: str, in_ch: int, input_hw: Tuple[int, int]) -> nn.Module:
    return MODELS[name](in_ch=in_ch, input_hw=input_hw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

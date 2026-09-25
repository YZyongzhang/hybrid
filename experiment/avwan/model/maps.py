"""Encode egocentric AVWAN occupancy/exploration and acoustic maps as tokens."""
import torch
from torch import nn


class MapEncoder(nn.Module):
    def __init__(self, channels, output_grid, large=False):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 5 if large else 3, stride=4 if large else 1,
                      padding=2 if large else 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((output_grid, output_grid)))

    def forward(self, value):
        return self.cnn(value.permute(0, 3, 1, 2).contiguous()).flatten(2).transpose(1, 2)


def unpack_geometry(value):
    """Expand losslessly packed cache data on its current device (normally GPU)."""
    if value.ndim == 4:
        return value.float()
    if value.ndim != 2 or value.shape[1] != 40000 or value.dtype != torch.uint8:
        raise ValueError('Expected packed gm [B,40000] uint8 or unpacked [B,400,400,2]')
    shifts = torch.arange(7, -1, -1, device=value.device, dtype=torch.uint8)
    return ((value.unsqueeze(-1) >> shifts) & 1).reshape(-1, 400, 400, 2).float()

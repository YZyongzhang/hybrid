import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Encoder import MultiHeadAttention


class MultiHeadAttentionLayer(MultiHeadAttention):
    """Attend over tokens within each sample, retaining the batch dimension."""
    def __init__(self, hidden_size, n_heads, dropout):
        super().__init__(hidden_size, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        if query.ndim == 2:
            query, key, value = (item.unsqueeze(1) for item in (query, key, value))
        return self.dropout(super().forward(query, key, value))


def layer_init(module):
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class AudioCRNN(nn.Module):
    def __init__(self, normalization='standard') -> None:
        super().__init__()
        if normalization not in ('standard', 'binaural_whiten', 'binaural_ratio',
                                 'binaural_robust_ratio'):
            raise ValueError(f'Unknown audio normalization: {normalization}')
        self.normalization = normalization
        self._number_input_audio = 2
        cnn_dimensions = np.array(
            (65,26), dtype=np.int32
        )

        if cnn_dimensions[0] < 30 or cnn_dimensions[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dimensions = (cnn_dimensions - np.array(kernel_size)) // np.array(stride) + 1
            
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._number_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            nn.Flatten(),
            nn.Linear(64 * cnn_dimensions[0] * cnn_dimensions[1], 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2)
        )
        self.attn = MultiHeadAttentionLayer(
            hidden_size=128,
            n_heads=16,
            dropout=0.05,
        )
        self.fnn = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 8),
        )
    
        layer_init(self.cnn)
        layer_init(self.fnn)

    def forward(self, audio):
        return self.fnn(self.encoder_forward(audio).squeeze(1))

    def encoder_forward(self, audio):
        audio = audio.float()
        if self.normalization == 'binaural_whiten':
            audio = binaural_whiten(audio)
        if self.normalization == 'binaural_ratio':
            # A fixed scale preserves the strength of interaural differences.
            # Per-sample standardization here would erase that spatial cue.
            audio = instantaneous_binaural_ratio(audio) * 5.
        elif self.normalization == 'binaural_robust_ratio':
            audio = robust_binaural_ratio(audio) * 5.
        else:
            dims = (1, 2, 3)
            audio = (audio - audio.mean(dims, keepdim=True)) / (audio.std(dims, keepdim=True, unbiased=False) + 1e-6)
        x = self.cnn(audio.permute(0, 3, 1, 2).contiguous())
        return self.attn(x, x, x)


def binaural_whiten(audio):
    """Remove shared source spectrum/gain, preserving left/right differences.

    Input is SoundSpaces log1p magnitude. Normalize both ears by the SAME
    time/ear-average magnitude at each frequency; independent ear normalization
    would erase localization cues. The relative floor suppresses silent bands.
    """
    magnitude = torch.expm1(audio.float().clamp(0, 20))
    reference = magnitude.mean((2, 3), keepdim=True)
    floor = reference.mean(1, keepdim=True) * .03
    return torch.log1p(magnitude / reference.clamp_min(floor).clamp_min(1e-12))


def instantaneous_binaural_ratio(audio):
    """Cancel common source coloration independently at every time/frequency bin."""
    magnitude = torch.expm1(audio.float().clamp(0,20))
    left, right = magnitude.unbind(-1)
    contrast = (left-right)/(left+right).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.stack((contrast,-contrast),dim=-1)


def robust_binaural_ratio(audio):
    """Source-scale invariant left/right contrast with a relative silence floor.

    The original two channels were exact negatives. Here the second channel
    carries a local energy-weighted contrast. Both channels retain direction,
    while low-energy bins contribute less instead of amplifying acoustic noise.
    A gain shared by both ears cancels, including in the relative floor.
    """
    magnitude = torch.expm1(audio.float().clamp(0, 20))
    left, right = magnitude.unbind(-1)
    total = left + right
    difference = left - right
    floor = total.mean((1, 2), keepdim=True) * .03
    raw = difference / (total + floor).clamp_min(1e-12)
    smooth_difference = F.avg_pool2d(difference.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    smooth_total = F.avg_pool2d(total.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    smooth = smooth_difference / (smooth_total + floor).clamp_min(1e-12)
    return torch.stack((raw, -smooth), dim=-1)

import numpy as np
import torch
import torch.nn as nn

from experiment.avnav.model.Encoder import MultiHeadAttention


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
    def __init__(self) -> None:
        super().__init__()
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
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        audio = audio.permute(0, 3, 1, 2)
        x = self.cnn(audio)
        x = self.attn.forward(x, x, x)
        x = x.squeeze(1)
        x = self.fnn(x)
        return x
    def encoder_forward(self, audio):
        audio = (audio - audio.mean()) / (audio.std() + 1e-6)
        audio = audio.permute(0, 3, 1, 2)
        x = self.cnn(audio)
        x = self.attn.forward(x, x, x)
        return x

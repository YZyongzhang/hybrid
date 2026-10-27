"""AVNav foundation with one RGB frame and no depth input."""

import torch
from torch import nn

from experiment.avnav.model.Encoder import Encoder
from experiment.avnav.model.audio import AudioCRNN
from experiment.avnav.model.vit import PositionalEcoder
from experiment.avnav_real.model import FinalModel


class RGBSingleFrameEncoder(nn.Module):
    """AVNav patch encoder receiving only the current RGB frame."""

    def __init__(self, output_dim=128, kernel_size=32):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(
            in_channels=3, out_channels=output_dim,
            kernel_size=kernel_size, stride=kernel_size,
        )

    def forward(self, rgb):
        return self.conv2d_1(rgb)


class NetworkSingleFrameNoDepth(nn.Module):
    """Same AVNav audio/transformer/fusion/action stack with current RGB only."""

    architecture = "avnav_audio_rgb_single_frame_no_depth"
    embedding_dim = 256
    visual_input_channels = 3

    def __init__(self):
        super().__init__()
        self.visual_encoder = RGBSingleFrameEncoder()
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder = Encoder(
            d_model=128, ffn_hidden=64, n_head=4, n_layers=3, drop_prob=0.2
        )
        self.audio_encoder = AudioCRNN()
        self.vaencoder = Encoder(
            d_model=128, ffn_hidden=64, n_head=4, n_layers=3, drop_prob=0.2
        )
        self.final = FinalModel(hidden_dim=256, output_dim=4)

    @staticmethod
    def _prepare_inputs(audio, rgb):
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(
                f"Expected single current RGB frame [B,H,W,3], got {tuple(rgb.shape)}"
            )
        return audio, rgb.permute(0, 3, 1, 2)

    def _action_embedding(self, audio, rgb):
        audio, rgb = self._prepare_inputs(audio, rgb)
        audio_dims = tuple(range(1, audio.ndim))
        audio = (audio - audio.mean(audio_dims, keepdim=True)) / (
            audio.std(audio_dims, keepdim=True, unbiased=False) + 1e-6
        )
        audio_tokens = self.audio_encoder.encoder_forward(audio)
        visual = self.visual_encoder(rgb)
        batch, channels, height, width = visual.shape
        visual = visual.reshape(batch, height * width, channels)
        visual = visual + self.add_position(visual)
        visual = self.visual_transformer_encoder(visual)
        fused = torch.cat((audio_tokens, visual), dim=1)
        fused = self.vaencoder(fused + self.add_position(fused))
        return self.final.fc1(fused)

    def embedding_forward(self, audio, rgb):
        return self._action_embedding(audio, rgb)

    def forward(self, audio, rgb):
        embedding = self._action_embedding(audio, rgb)
        dropped = self.final.dropout(embedding)
        return (
            self.final.polar_head_probe(dropped),
            self.final.angle_head_probe(dropped),
            self.final.action_head(dropped),
        )

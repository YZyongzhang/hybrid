"""Sample-local input transforms shared by training and inference."""
import torch


def normalize_audio(audio):
    """Normalize each spectrogram jointly across both ears, never across samples."""
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4 or audio.shape[-1] != 2:
        raise ValueError(f"Expected [B,F,T,2], got {tuple(audio.shape)}")
    dims = (1, 2, 3)
    mean = audio.mean(dim=dims, keepdim=True)
    std = audio.std(dim=dims, keepdim=True, unbiased=False)
    return (audio - mean) / std.clamp_min(1e-6)


def spatial_tokens(features):
    """One token per spatial cell, retaining its complete channel vector."""
    return features.flatten(2).transpose(1, 2).contiguous()

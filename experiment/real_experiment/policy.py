"""Inference for the validation-selected AVNav RGB + binaural audio pair."""

from collections import deque
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .models.foundation import Network as HybridFoundation
from .models.offline_sac_v1_5 import SAC_LSTM_CQL_v1_5

ArrayLike = Union[np.ndarray, torch.Tensor]
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FOUNDATION = PACKAGE_DIR / "checkpoints/foundation_rgb_best.pth"
DEFAULT_POLICY = PACKAGE_DIR / "checkpoints/corrective_epoch_015_best_policy.pt"
DEFAULT_CONFIG = PACKAGE_DIR / "checkpoints/training_config.json"


class AVNavRealPolicy:
    """Encode two RGB frames and audio, then return the raw SAC-LSTM action.

    Call ``reset`` between episodes. Optional legacy depth arguments are ignored.
    """

    def __init__(
        self,
        foundation_checkpoint: Union[str, Path] = DEFAULT_FOUNDATION,
        policy_checkpoint: Union[str, Path] = DEFAULT_POLICY,
        device: Optional[Union[str, torch.device]] = None,
        deterministic: bool = True,
        sequence_length: int = 5,
        training_config: Union[str, Path] = DEFAULT_CONFIG,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        cfg = json.loads(Path(training_config).read_text())
        if not deterministic or sequence_length != cfg["sequence_length"]:
            raise ValueError("The frozen protocol requires deterministic actions and five-step history")
        self.deterministic = True
        self.sequence_length = int(cfg["sequence_length"])
        self.state_dim = int(cfg["state_dim"])

        self.foundation = HybridFoundation().to(self.device)
        self.foundation.load_state_dict(torch.load(
            foundation_checkpoint, map_location=self.device, weights_only=True
        ), strict=True)
        self.foundation.eval().requires_grad_(False)

        self.policy = SAC_LSTM_CQL_v1_5(
            state_dim=self.state_dim,
            hidden_dim=int(cfg["hidden_dim"]), action_dim=4,
            actor_lr=float(cfg["actor_lr"]), critic_lr=float(cfg["critic_lr"]),
            alpha_lr=float(cfg["alpha_lr"]), target_entropy=float(cfg["target_entropy"]),
            tau=float(cfg["target_tau"]), gamma=float(cfg["gamma"]),
            beta=float(cfg["cql_weight"]), device=self.device,
            lstm_hidden_dim=int(cfg["lstm_hidden_dim"]), lstm_num_layers=1,
        ).to(self.device)
        self.policy.load_state_dict(torch.load(
            policy_checkpoint, map_location=self.device, weights_only=True
        ), strict=True)
        self.policy.eval().requires_grad_(False)
        self.reset()

    def reset(self) -> None:
        """Clear frame and embedding history at the start of an episode."""
        self._previous_rgb = None
        self._states = deque(maxlen=self.sequence_length)

    @staticmethod
    def _rgb_tensor(rgb: ArrayLike) -> torch.Tensor:
        source = torch.as_tensor(rgb).detach()
        tensor = source.float()
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"rgb must be HWC with 3 channels, got {tuple(tensor.shape)}")
        if not source.is_floating_point() or (tensor.numel() and tensor.max().item() > 1.5):
            tensor = tensor / 255.0
        return tensor

    @staticmethod
    def _audio_tensor(audio: ArrayLike) -> torch.Tensor:
        tensor = torch.as_tensor(audio).detach().float()
        if tensor.ndim != 3 or tensor.shape[-1] != 2:
            raise ValueError(f"audio spectrogram must be [F, T, 2], got {tuple(tensor.shape)}")
        return tensor

    @torch.inference_mode()
    def act(self, audio: ArrayLike, rgb: ArrayLike, depth=None) -> int:
        """Select a raw action using only audio and RGB."""
        audio_tensor = self._audio_tensor(audio).to(self.device)
        rgb_tensor = self._rgb_tensor(rgb).to(self.device)
        previous_rgb = (
            torch.zeros_like(rgb_tensor) if self._previous_rgb is None else self._previous_rgb
        )
        rgb_pair = torch.cat((previous_rgb, rgb_tensor), dim=-1).unsqueeze(0)
        embedding = self.foundation.embedding_forward(
            audio_tensor.unsqueeze(0), rgb_pair
        ).float().reshape(-1)
        if embedding.numel() != self.state_dim:
            raise ValueError(f"foundation embedding must contain {self.state_dim} values")
        self._states.append(embedding)
        sequence = embedding.new_zeros((self.sequence_length, self.state_dim))
        sequence[-len(self._states):] = torch.stack(list(self._states))
        action = self.policy.get_action(sequence.unsqueeze(0), eval=True)
        self._previous_rgb = rgb_tensor.clone()
        return int(action)

    def act_observation(self, observation: Dict[str, Any]) -> int:
        """Accept direct, single-batch, or simulator tuple spectrograms."""
        audio = observation["spectrogram"]
        if isinstance(audio, tuple) and len(audio) == 2:
            audio = audio[0]
        if hasattr(audio, "ndim") and audio.ndim == 4 and audio.shape[0] == 1:
            audio = audio[0]
        return self.act(audio, observation["rgb"])

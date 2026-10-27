"""Inference wrapper for the paired AVNav RGB-D + audio checkpoints."""

from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Union

import numpy as np
import torch

from .models.foundation import Network as HybridFoundation
from .models.offline_sac_v1_5 import SAC_LSTM_CQL_v1_5

ArrayLike = Union[np.ndarray, torch.Tensor]


class AVNavRealPolicy:
    """Run the bundled foundation encoder and SAC-LSTM policy.

    ``act_observation`` expects an AVNav-style observation containing ``rgb``,
    ``depth`` and ``spectrogram``. Call ``reset`` between episodes. Returned
    integers are the original four discrete AVNav action IDs; this wrapper does
    not alter or post-process policy actions.
    """

    def __init__(
        self,
        foundation_checkpoint: Union[str, Path],
        policy_checkpoint: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        deterministic: bool = True,
        sequence_length: int = 5,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.deterministic = bool(deterministic)
        self.sequence_length = int(sequence_length)
        self.state_dim = 256

        self.foundation = HybridFoundation().to(self.device)
        foundation_state = torch.load(foundation_checkpoint, map_location=self.device)
        self.foundation.load_state_dict(foundation_state)
        self.foundation.eval()

        self.policy = SAC_LSTM_CQL_v1_5(
            state_dim=256,
            hidden_dim=128,
            action_dim=4,
            actor_lr=2e-5,
            critic_lr=5e-5,
            alpha_lr=1e-5,
            target_entropy=1.08,
            tau=0.005,
            gamma=0.75,
            beta=0.5,
            device=self.device,
            lstm_hidden_dim=256,
            lstm_num_layers=1,
        ).to(self.device)
        policy_state = torch.load(policy_checkpoint, map_location=self.device)
        # Match the archived evaluator's checkpoint-loading behavior.
        self.policy.load_state_dict(policy_state, strict=False)
        self.policy.eval()

        self.reset()

    def reset(self) -> None:
        """Clear two-frame and five-step history at the start of an episode."""
        self._previous_rgb: Optional[torch.Tensor] = None
        self._previous_depth: Optional[torch.Tensor] = None
        self._states: Deque[torch.Tensor] = deque(maxlen=self.sequence_length)

    @staticmethod
    def _rgb_tensor(rgb: ArrayLike) -> torch.Tensor:
        tensor = torch.as_tensor(rgb).detach().float()
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"rgb must be HWC with 3 channels, got {tuple(tensor.shape)}")
        if tensor.numel() and tensor.max().item() > 1.5:
            tensor = tensor / 255.0
        return tensor

    @staticmethod
    def _depth_tensor(depth: ArrayLike) -> torch.Tensor:
        tensor = torch.as_tensor(depth).detach().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim != 3 or tensor.shape[-1] != 1:
            raise ValueError(f"depth must be HW or HWC with 1 channel, got {tuple(tensor.shape)}")
        return tensor

    @staticmethod
    def _audio_tensor(audio: ArrayLike) -> torch.Tensor:
        tensor = torch.as_tensor(audio).detach().float()
        if tensor.ndim != 3:
            raise ValueError(f"audio spectrogram must be [F, T, 2], got {tuple(tensor.shape)}")
        return tensor

    @torch.inference_mode()
    def act(self, audio: ArrayLike, rgb: ArrayLike, depth: ArrayLike) -> int:
        """Select one action from current audio spectrogram, RGB and depth."""
        audio_tensor = self._audio_tensor(audio)
        rgb_tensor = self._rgb_tensor(rgb)
        depth_tensor = self._depth_tensor(depth)
        if rgb_tensor.shape[:2] != depth_tensor.shape[:2]:
            raise ValueError("rgb and depth spatial dimensions must match")

        if self._previous_rgb is None:
            previous_rgb = torch.zeros_like(rgb_tensor)
            previous_depth = torch.zeros_like(depth_tensor)
        else:
            previous_rgb = self._previous_rgb
            previous_depth = self._previous_depth

        # Keep the archived inference layout: previous and current frames joined
        # along image width, with channels remaining in HWC order.
        two_frame_rgb = torch.cat((previous_rgb, rgb_tensor), dim=1)
        two_frame_depth = torch.cat((previous_depth, depth_tensor), dim=1)
        embedding = self.foundation.embedding_forward(
            audio_tensor.to(self.device),
            two_frame_rgb.to(self.device),
            two_frame_depth.to(self.device),
        ).detach().reshape(-1)
        if embedding.numel() != self.state_dim:
            raise ValueError(
                f"foundation embedding has {embedding.numel()} values; expected {self.state_dim}"
            )

        self._states.append(embedding)
        history = list(self._states)
        while len(history) < self.sequence_length:
            history.insert(0, torch.zeros_like(embedding))
        state_sequence = torch.stack(history, dim=0).unsqueeze(0)
        action = self.policy.get_action(
            state_sequence.to(self.device), eval=self.deterministic
        )

        self._previous_rgb = rgb_tensor
        self._previous_depth = depth_tensor
        return int(action)

    def act_observation(self, observation: Dict[str, Any]) -> int:
        """Select from an AVNav observation dictionary."""
        audio = observation["spectrogram"][0]
        return self.act(audio, observation["rgb"], observation["depth"])

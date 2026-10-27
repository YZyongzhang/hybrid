from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_ANGLE_BINS = 8


def _group_count(num_channels: int) -> int:
    for groups in (16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _make_transformer_encoder(
    d_model: int,
    nhead: int,
    num_layers: int,
    dropout: float,
    dim_feedforward: int,
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class ConvTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 128,
        hidden_channels: int = 32,
        dropout: float = 0.1,
        pool_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        mid_channels = hidden_channels * 2
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.GELU(),
        ]
        if pool_size is not None:
            layers.append(nn.AdaptiveAvgPool2d(pool_size))
        layers.append(nn.Dropout2d(dropout))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x.float())
        return feat.flatten(2).transpose(1, 2)


class SpatialAudioDecomposer(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def _normalize_map(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio.dim() == 3:
            audio = audio.unsqueeze(0)
        if audio.dim() != 4 or audio.size(-1) != 2:
            raise ValueError(
                f"audio must have shape [B, F, T, 2] or [F, T, 2], got {tuple(audio.shape)}"
            )

        left = audio[..., 0].float()
        right = audio[..., 1].float()

        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)

        left_norm = self._normalize_map(left)
        right_norm = self._normalize_map(right)
        side_norm = self._normalize_map(side)
        mid_norm = self._normalize_map(mid)

        ild = torch.log(left.abs() + self.eps) - torch.log(right.abs() + self.eps)
        ild = self._normalize_map(ild)

        side_ratio = side / (mid.abs() + self.eps)
        side_ratio = self._normalize_map(side_ratio)

        mono_energy = torch.log(0.5 * (left.abs() + right.abs()) + self.eps)
        mono_energy = self._normalize_map(mono_energy)

        spatial_map = torch.stack(
            [left_norm, right_norm, side_norm, ild, side_ratio], dim=1
        )
        timbre_map = torch.stack([mid_norm, mono_energy], dim=1)
        return spatial_map, timbre_map


class TimbreInvariantAudioEncoder(nn.Module):
    """
    Spatial branch keeps binaural geometry.
    Timbre branch models source identity / coloration and is only used as a nuisance signal
    that gets projected out from the spatial branch.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.decomposer = SpatialAudioDecomposer()
        self.position = SinusoidalPositionalEncoding(embed_dim, max_len=256)

        self.spatial_tokenizer = ConvTokenizer(
            in_channels=5,
            embed_dim=embed_dim,
            hidden_channels=32,
            dropout=dropout,
            pool_size=(6, 4),
        )
        self.timbre_tokenizer = ConvTokenizer(
            in_channels=2,
            embed_dim=embed_dim,
            hidden_channels=24,
            dropout=dropout,
            pool_size=(4, 3),
        )

        self.spatial_encoder = _make_transformer_encoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
        )
        self.timbre_encoder = _make_transformer_encoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=max(1, num_layers - 1),
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
        )
        self.timbre_to_spatial = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.timbre_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.refine_encoder = _make_transformer_encoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=1,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        spatial_map, timbre_map = self.decomposer(audio)

        spatial_tokens = self.spatial_tokenizer(spatial_map)
        timbre_tokens = self.timbre_tokenizer(timbre_map)

        spatial_tokens = self.spatial_encoder(spatial_tokens + self.position(spatial_tokens))
        timbre_tokens = self.timbre_encoder(timbre_tokens + self.position(timbre_tokens))

        timbre_residual, _ = self.timbre_to_spatial(
            spatial_tokens, timbre_tokens, timbre_tokens
        )
        timbre_feature = timbre_tokens.mean(dim=1)
        timbre_basis = F.normalize(self.timbre_projection(timbre_feature), dim=-1).unsqueeze(1)
        shared_projection = (spatial_tokens * timbre_basis).sum(dim=-1, keepdim=True) * timbre_basis

        spatial_tokens = self.output_norm(
            spatial_tokens - self.residual_scale * timbre_residual - shared_projection
        )
        spatial_tokens = self.refine_encoder(spatial_tokens + self.position(spatial_tokens))
        spatial_feature = spatial_tokens.mean(dim=1)

        orthogonality_loss = (
            F.normalize(spatial_feature, dim=-1) * F.normalize(timbre_feature, dim=-1)
        ).sum(dim=-1).pow(2).mean()

        return {
            "spatial_tokens": spatial_tokens,
            "timbre_tokens": timbre_tokens,
            "spatial_feature": spatial_feature,
            "timbre_feature": timbre_feature,
            "orthogonality_loss": orthogonality_loss,
        }


class VisualTokenEncoder(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        hidden_channels = 32
        mid_channels = hidden_channels * 2
        first_conv: nn.Module
        if in_channels is None:
            first_conv = nn.LazyConv2d(
                hidden_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            )
        else:
            first_conv = nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            )
        self.tokenizer = nn.Sequential(
            first_conv,
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels // 2, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(mid_channels), mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(embed_dim), embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Dropout2d(dropout),
        )
        self.position = SinusoidalPositionalEncoding(embed_dim, max_len=256)
        self.encoder = _make_transformer_encoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
        )

    def forward(self, rgbd: torch.Tensor) -> torch.Tensor:
        if self.in_channels is not None and rgbd.size(1) != self.in_channels:
            raise ValueError(
                f"expected visual channels={self.in_channels}, got {rgbd.size(1)}. "
                "If depth/rgb channels differ, pass the correct visual_in_channels when constructing Network."
            )
        visual_feat = self.tokenizer(rgbd.float())
        visual_tokens = visual_feat.flatten(2).transpose(1, 2)
        return self.encoder(visual_tokens + self.position(visual_tokens))


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.audio_queries_visual = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.visual_queries_audio = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_norm = nn.LayerNorm(embed_dim)
        self.visual_norm = nn.LayerNorm(embed_dim)
        self.position = SinusoidalPositionalEncoding(embed_dim, max_len=256)
        self.shared_encoder = _make_transformer_encoder(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
        )

    def forward(self, audio_tokens: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        audio_context, _ = self.audio_queries_visual(audio_tokens, visual_tokens, visual_tokens)
        audio_tokens = self.audio_norm(audio_tokens + audio_context)

        visual_context, _ = self.visual_queries_audio(visual_tokens, audio_tokens, audio_tokens)
        visual_tokens = self.visual_norm(visual_tokens + visual_context)

        fused_tokens = torch.cat([audio_tokens, visual_tokens], dim=1)
        return self.shared_encoder(fused_tokens + self.position(fused_tokens))


class Network(nn.Module):
    """
    Hybrid model for better unheard generalization.

    Key idea:
    1. Decompose binaural audio into a spatial branch and a timbre branch.
    2. Use the timbre branch only to estimate nuisance information and remove it.
    3. Fuse purified spatial tokens with RGBD so action policy depends more on direction than on source identity.
    """

    def __init__(
        self,
        action_dim: int = 4,
        pose_dim: int = 4,
        angle_bins: int = NUM_ANGLE_BINS,
        modality_dim: int = 128,
        hidden_dim: int = 256,
        visual_in_channels: Optional[int] = None,
        ego_map_channels: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.angle_bins = angle_bins
        self.visual_in_channels = visual_in_channels
        self.ego_map_channels = ego_map_channels
        self.device = torch.device('cuda')

        self.audio_encoder = TimbreInvariantAudioEncoder(
            embed_dim=modality_dim,
            num_heads=4,
            dropout=dropout,
            num_layers=2,
        )
        self.visual_encoder = VisualTokenEncoder(
            in_channels=visual_in_channels,
            embed_dim=modality_dim,
            num_heads=4,
            dropout=dropout,
            num_layers=2,
        )
        self.av_fusion = CrossModalFusion(
            embed_dim=modality_dim,
            num_heads=4,
            dropout=dropout,
            num_layers=2,
        )

        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, modality_dim),
            nn.GELU(),
            nn.LayerNorm(modality_dim),
        )
        self.ego_map_encoder = nn.Sequential(
            nn.Conv2d(ego_map_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(64), 64),
            nn.GELU(),
            nn.Conv2d(64, modality_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(modality_dim), modality_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(modality_dim),
        )

        self.state_fusion = nn.Sequential(
            nn.Linear(modality_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.polar_head = nn.Linear(modality_dim, 2)
        self.angle_head = nn.Sequential(
            nn.Linear(modality_dim, modality_dim),
            nn.GELU(),
            nn.LayerNorm(modality_dim),
            nn.Linear(modality_dim, angle_bins),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
        )

    def _prepare_audio(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.unsqueeze(0)
        return audio.float()

    def _prepare_rgbd(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)
        if depth.dim() == 3:
            depth = depth.unsqueeze(0)

        rgb = rgb.float()
        depth = depth.float()
        if rgb.max() > 1.5:
            rgb = rgb / 255.0

        rgb = rgb.permute(0, 3, 1, 2)
        depth = depth.permute(0, 3, 1, 2)
        return torch.cat([rgb, depth], dim=1)

    def _encode_pose(self, pose: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if pose is None:
            return torch.zeros(batch_size, self.pose_encoder[0].out_features, device=device)
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        return self.pose_encoder(pose.float().to(device))

    def _encode_ego_map(
        self,
        ego_map: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if ego_map is None:
            return torch.zeros(batch_size, self.angle_head[0].in_features, device=device)

        if ego_map.dim() == 3:
            ego_map = ego_map.unsqueeze(0)
        ego_map = ego_map.float().to(device)
        if ego_map.size(-1) == self.ego_map_channels:
            ego_map = ego_map.permute(0, 3, 1, 2)
        elif ego_map.size(1) != self.ego_map_channels:
            raise ValueError(
                f"ego_map must have {self.ego_map_channels} channels in first or last dim, got {tuple(ego_map.shape)}"
            )
        return self.ego_map_encoder(ego_map)

    def encode_state(
        self,
        audio: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        ego_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        audio = self._prepare_audio(audio)
        rgbd = self._prepare_rgbd(rgb, depth).to(audio.device)

        audio_dict = self.audio_encoder(audio)
        visual_tokens = self.visual_encoder(rgbd)
        fused_tokens = self.av_fusion(audio_dict["spatial_tokens"], visual_tokens)
        fused_feature = fused_tokens.mean(dim=1)

        pose_feature = self._encode_pose(pose, audio.size(0), audio.device)
        ego_map_feature = self._encode_ego_map(ego_map, audio.size(0), audio.device)

        state_feature = self.state_fusion(
            torch.cat(
                [
                    audio_dict["spatial_feature"],
                    fused_feature,
                    pose_feature,
                    ego_map_feature,
                ],
                dim=1,
            )
        )
        return {
            "state_feature": state_feature,
            "spatial_feature": audio_dict["spatial_feature"],
            "timbre_feature": audio_dict["timbre_feature"],
            "fused_feature": fused_feature,
            "orthogonality_loss": audio_dict["orthogonality_loss"],
        }

    def embedding_forward(
        self,
        audio: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        ego_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedding = self.encode_state(audio, rgb, depth, pose=pose, ego_map=ego_map)["state_feature"]
        return embedding.squeeze(0)

    def forward_joint(
        self,
        audio: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        ego_map: Optional[torch.Tensor] = None,
        *,
        return_aux: bool = False,
    ):
        polar_predict, angle_logits, action_logits, aux = self.forward(
            audio,
            rgb,
            depth,
            pose=pose,
            ego_map=ego_map,
            return_aux=True,
        )
        if return_aux:
            return action_logits, angle_logits, aux
        return action_logits, angle_logits

    def forward(
        self,
        audio: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        ego_map: Optional[torch.Tensor] = None,
        *,
        return_aux: bool = False,
    ):
        encoded = self.encode_state(audio, rgb, depth, pose=pose, ego_map=ego_map)
        polar_predict = self.polar_head(encoded["spatial_feature"])
        angle_logits = self.angle_head(encoded["spatial_feature"])
        action_logits = self.action_head(encoded["state_feature"])

        if return_aux:
            aux = {
                "orthogonality_loss": encoded["orthogonality_loss"],
                "spatial_feature": encoded["spatial_feature"],
                "timbre_feature": encoded["timbre_feature"],
                "fused_feature": encoded["fused_feature"],
            }
            return polar_predict, angle_logits, action_logits, aux
        return polar_predict, angle_logits, action_logits

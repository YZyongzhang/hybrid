import sys
import torch
import torch.nn as nn
from .vit import ViTEncoder , VisualEncoder , PositionalEcoder
from .audio import AudioCRNN
from .Encoder import Encoder
from .maps import MapEncoder
class Attention(nn.Module):
    def __init__(self , input_dim , visual_dim , audio_dim ,hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)


        self.attention_weight = nn.Parameter(torch.randn(self.output_dim) * 0.01)
        
        
    def forward(self, encode):
        x1 = torch.relu(self.fc1(encode))
        x2 = torch.relu(self.fc2(x1))
        self.attention_weight.unsqueeze(1)
        # import pdb;pdb.set_trace()
        # y = torch.matmul(x2.unsqueeze(-1) , self.attention_weight)
        y = x2 * self.attention_weight
        return x2
    
class Vote(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.output_dim = output_dim
        
        self.voter = nn.Parameter(torch.randn(self.input_dim , 32) * 0.01)

    def forward(self,embeding):
        output = torch.matmul(embeding , self.voter)
        return output
    
class Decision(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(Decision, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_matrix):
        x = input_matrix.squeeze(1)
        # import pdb;pdb.set_trace()
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Finnal_model(nn.Module):
    def __init__(self, input_dim , hidden_dim , output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim , self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        
        # self.polar_head = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 128),
        #     nn.ReLU(),
        #     nn.LayerNorm(128),
        #     nn.Linear(128, 2)
        # )
        self.polar_head_probe = nn.Sequential(
            nn.Linear(self.hidden_dim , 2 )
        )

        # self.angle_head = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 128),
        #     nn.ReLU(),
        #     nn.LayerNorm(128),
        #     nn.Linear(128, 8)
        # )
        self.angle_head_probe = nn.Sequential(
            nn.Linear(self.hidden_dim , 8)
        )
        
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim)
        )

    def forward(self,encoder):
        x1 = self.fc1(encoder)
        x2 = self.dropout(x1)
        polar = self.polar_head_probe(x2)
        action_logits = self.action_head(x2)
        return polar, action_logits

class Network(nn.Module):
    """User hybrid architecture with egocentric map and valid-waypoint tokens."""
    def __init__(self, action_count=81, audio_normalization='standard', map_normalization='absolute', spatial_detail=False,
                 temporal_audio=False):
        super().__init__()
        if map_normalization not in ('absolute', 'relative'):
            raise ValueError(f'Unknown map normalization: {map_normalization}')
        self.map_normalization = map_normalization
        self.action_count = action_count
        self.visual_encoder = VisualEncoder()
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder = Encoder(128, 64, 4, 3, .2)
        self.audio_encoder = AudioCRNN(audio_normalization, temporal_context=temporal_audio)
        self.geometric_encoder = MapEncoder(2, output_grid=4, large=True)
        self.acoustic_encoder = MapEncoder(1, output_grid=2)
        self.mask_encoder = nn.Sequential(nn.Linear(action_count, 128), nn.ReLU())
        self.modality_embedding = nn.Parameter(torch.randn(5, 1, 128) * .01)
        self.vaencoder = Encoder(128, 64, 4, 3, .2)
        # 1 audio + 16 visual + 16 geometry + 4 acoustic + 1 mask token.
        self.final = Finnal_model(input_dim=38 * 128, hidden_dim=256, output_dim=action_count)
        self.detail_encoder = None
        if spatial_detail:
            if action_count != 81:
                raise ValueError('Spatial detail branch requires the 9x9 waypoint grid')
            # Retain every 1m acoustic cell and local 1m geometry cell rather
            # than representing the whole acoustic map with only four tokens.
            self.detail_encoder = nn.Sequential(nn.Linear(400 + 162 + 81 + 128 + 6, 256),
                                                nn.LayerNorm(256), nn.ReLU(), nn.Linear(256, 256))
            nn.init.zeros_(self.detail_encoder[-1].weight)
            nn.init.zeros_(self.detail_encoder[-1].bias)

    def _encode(self, audio, rgb, depth, gm, am, action_mask):
        if audio.ndim == 3:
            audio, rgb, depth, gm, am, action_mask = (
                value.unsqueeze(0) for value in (audio, rgb, depth, gm, am, action_mask))
        expected = ((65, 26, 2), (128, 128, 6), (128, 128, 2), (400, 400, 2), (20, 20, 1))
        if any(tuple(value.shape[1:]) != shape for value, shape in zip((audio, rgb, depth, gm, am), expected)):
            raise ValueError('Expected audio [65,26,2], two-frame RGB/depth [128,128,6/2], gm [400,400,2], am [20,20,1]')
        if action_mask.shape != (audio.shape[0], self.action_count) or not action_mask.bool().any(dim=1).all():
            raise ValueError('Every observation needs a nonempty waypoint action mask')
        rgbd = torch.cat((rgb, depth), dim=-1).permute(0, 3, 1, 2).contiguous()
        audio_token = self.audio_encoder.encoder_forward(audio)
        visual = self.visual_encoder(rgbd).flatten(2).transpose(1, 2)
        visual = self.visual_transformer_encoder(visual + self.add_position(visual))
        geometry = self.geometric_encoder(gm.float())
        # Dynamic range compression preserves zero and the sign of acoustic values.
        am = am.float()
        if self.map_normalization == 'relative':
            # Preserve visited-cell energy ratios and zero/unobserved cells.
            am = 10. * am / am.abs().amax((1, 2, 3), keepdim=True).clamp_min(1e-12)
        acoustic = self.acoustic_encoder(am.sign() * torch.log1p(am.abs()))
        valid = self.mask_encoder(action_mask.float()).unsqueeze(1)
        groups = (audio_token, visual, geometry, acoustic, valid)
        tokens = torch.cat([group + self.modality_embedding[i] for i, group in enumerate(groups)], dim=1)
        fused = self.vaencoder(tokens + self.add_position(tokens))
        embedding = self.final.fc1(fused)
        if self.detail_encoder is not None:
            local = gm.float().permute(0, 3, 1, 2)[:, :, 155:245, 155:245]
            local = nn.functional.avg_pool2d(local, 10).flatten(1)
            energy = am.clamp_min(0).flatten(1)
            relative = energy / energy.amax(1, keepdim=True).clamp_min(1e-12)
            center = relative[:, 210]  # egocentric acoustic map center=(10,10)
            others = relative.clone();others[:, 210] = 0
            known = (energy > 0).sum(1).float()
            peak = relative.argmax(1)
            peak_distance = (((peak//20-10).float().square() + (peak%20-10).float().square()).sqrt()/14.2)
            peak_distance = peak_distance * (known > 0)
            neighborhood = relative.reshape(-1,20,20)[:,9:12,9:12]
            neighbor_count = (neighborhood > 0).sum((1,2)).float()-(center > 0).float()
            neighbor_mean = (neighborhood.sum((1,2))-center)/neighbor_count.clamp_min(1)
            statistics = torch.stack((center, torch.log1p(center/others.amax(1).clamp_min(.01)),
                                      torch.log1p(known)/6., relative.sum(1)/known.clamp_min(1),
                                      peak_distance, torch.log1p(center/neighbor_mean.clamp_min(.01))), dim=1)
            detail = torch.cat((torch.log1p(am.clamp_min(0)).flatten(1), local,
                                action_mask.float(), audio_token.squeeze(1), statistics), dim=1)
            embedding = embedding + self.detail_encoder(detail)
        return embedding, audio_token

    def forward_joint(self, audio, rgb, depth, gm, am, action_mask):
        embedding, audio_token = self._encode(audio, rgb, depth, gm, am, action_mask)
        actions = self.final.action_head(self.final.dropout(embedding)).float()
        actions = actions.masked_fill(~action_mask.bool(), -1e9)
        angles = self.audio_encoder.fnn(audio_token.squeeze(1)).float()
        return actions, angles

    def forward_joint_features(self, audio, rgb, depth, gm, am, action_mask,
                               return_audio=False):
        """Expose fused features for source-view consistency training."""
        embedding, audio_token = self._encode(audio, rgb, depth, gm, am, action_mask)
        actions = self.final.action_head(self.final.dropout(embedding)).float()
        actions = actions.masked_fill(~action_mask.bool(), -1e9)
        angles = self.audio_encoder.fnn(audio_token.squeeze(1)).float()
        if return_audio:
            return actions, angles, embedding, audio_token.squeeze(1)
        return actions, angles, embedding

    def forward(self, audio, rgb, depth, gm, am, action_mask):
        embedding, _ = self._encode(audio, rgb, depth, gm, am, action_mask)
        dropped = self.final.dropout(embedding)
        actions = self.final.action_head(dropped).float().masked_fill(~action_mask.bool(), -1e9)
        return self.final.polar_head_probe(dropped), self.final.angle_head_probe(dropped), actions

    def embedding_forward(self, audio, rgb, depth, gm, am, action_mask):
        """Return fused [B,256] features, including map history, for downstream RL."""
        return self._encode(audio, rgb, depth, gm, am, action_mask)[0]

    def embedding_forward_attention(self, audio, rgb, depth, gm, am, action_mask):
        embedding, audio_token = self._encode(audio, rgb, depth, gm, am, action_mask)
        return audio_token.squeeze(1), embedding

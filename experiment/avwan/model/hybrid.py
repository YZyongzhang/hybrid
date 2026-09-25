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
    def __init__(self, action_count=81):
        super().__init__()
        self.action_count = action_count
        self.visual_encoder = VisualEncoder()
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder = Encoder(128, 64, 4, 3, .2)
        self.audio_encoder = AudioCRNN()
        self.geometric_encoder = MapEncoder(2, output_grid=4, large=True)
        self.acoustic_encoder = MapEncoder(1, output_grid=2)
        self.mask_encoder = nn.Sequential(nn.Linear(action_count, 128), nn.ReLU())
        self.modality_embedding = nn.Parameter(torch.randn(5, 1, 128) * .01)
        self.vaencoder = Encoder(128, 64, 4, 3, .2)
        # 1 audio + 16 visual + 16 geometry + 4 acoustic + 1 mask token.
        self.final = Finnal_model(input_dim=38 * 128, hidden_dim=256, output_dim=action_count)

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
        acoustic = self.acoustic_encoder(am.float().sign() * torch.log1p(am.float().abs()))
        valid = self.mask_encoder(action_mask.float()).unsqueeze(1)
        groups = (audio_token, visual, geometry, acoustic, valid)
        tokens = torch.cat([group + self.modality_embedding[i] for i, group in enumerate(groups)], dim=1)
        fused = self.vaencoder(tokens + self.add_position(tokens))
        return self.final.fc1(fused), audio_token

    def forward_joint(self, audio, rgb, depth, gm, am, action_mask):
        embedding, audio_token = self._encode(audio, rgb, depth, gm, am, action_mask)
        actions = self.final.action_head(self.final.dropout(embedding)).float()
        actions = actions.masked_fill(~action_mask.bool(), -1e9)
        angles = self.audio_encoder.fnn(audio_token.squeeze(1)).float()
        return actions, angles

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

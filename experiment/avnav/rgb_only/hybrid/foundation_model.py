import sys
import torch
import torch.nn as nn
from .ViT import ViTEncoder
from .ViT import VisualEncoder , PositionalEcoder
from .Encoder import Encoder
from .preprocessing import spatial_tokens
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
        self.input_dim = 16*128
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
        polar = self.polar_head(x2)
        action_logits = self.action_head(x2)
        return polar, action_logits

class Network(nn.Module):
    """RGB-only ablation: original transformer/heads, with six RGB channels.

    Input is previous+current RGB [B,128,128,6], scaled to [0,1].
    No depth, audio, goal, reward, collision signal or privileged labels are inputs.
    """
    def __init__(self):
        super().__init__()
        self.visual_encoder = VisualEncoder(input_dim=6)
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder = Encoder(d_model=128, ffn_hidden=64, n_head=4, n_layers=3, drop_prob=0.2)
        self.vaencoder = Encoder(d_model=128, ffn_hidden=64, n_head=4, n_layers=3, drop_prob=0.2)
        self.final = Finnal_model(input_dim=512, hidden_dim=256, output_dim=4)

    def _action_embedding(self, rgb):
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (128,128,6):
            raise ValueError(f"Expected [B,128,128,6], got {tuple(rgb.shape)}")
        visual = spatial_tokens(self.visual_encoder(rgb.permute(0,3,1,2)))
        visual = self.visual_transformer_encoder(visual + self.add_position(visual))
        shared = self.vaencoder(visual + self.add_position(visual))
        return self.final.fc1(shared)

    def forward(self, rgb):
        dropped = self.final.dropout(self._action_embedding(rgb))
        return self.final.polar_head_probe(dropped), self.final.angle_head_probe(dropped), self.final.action_head(dropped)

    def embedding_forward(self, rgb):
        return self._action_embedding(rgb)

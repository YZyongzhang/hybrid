import sys
import torch
import torch.nn as nn
from .ViT import ViTEncoder
from .audio import AudioCRNN
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
        self.input_dim = 17*128
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
    def __init__(self):
        super().__init__()
        # self.vitpartnet = ViTEncoder()
        self.visual_encoder = VisualEncoder(input_dim=6)
        self.add_position = PositionalEcoder()
        self.visual_transformer_encoder =  Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        self.audio_encoder = AudioCRNN()

        self.vaencoder =  Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        self.final = Finnal_model(input_dim = 512 , hidden_dim= 256 , output_dim=4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    def _prepare_inputs(self, audio, rgb, depth=None):
        # Compatibility argument depth is never read; only two RGB frames enter vision.
        if audio.ndim == 3:
            audio = audio.unsqueeze(0)
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        if rgb.shape[-1] != 6:
            raise ValueError(f"Expected two RGB frames [B,H,W,6], got {tuple(rgb.shape)}")
        return audio, rgb.permute(0, 3, 1, 2)

    def _action_embedding(self, audio, rgb, depth=None):
        audio, rgbd = self._prepare_inputs(audio, rgb, depth)
        
        audio_encoder = self.audio_encoder.encoder_forward(audio)
        visual_cnn = self.visual_encoder(rgbd)
        v_batch , v_dim , v_h , v_w = visual_cnn.shape
        visual_cnn = spatial_tokens(visual_cnn)
        v_position = self.add_position(visual_cnn)
        visual_cnn = visual_cnn + v_position
        visual_encoder = self.visual_transformer_encoder(visual_cnn)
        concat_encoder = torch.cat((audio_encoder , visual_encoder ) , dim=1)
        p_share_encoder = self.add_position(concat_encoder)
        
        share_visual_audio_encoder = self.vaencoder(concat_encoder + p_share_encoder)
        embedding = self.final.fc1(share_visual_audio_encoder)
        return embedding

    def forward(self,audio , rgb , depth=None):
        embedding = self._action_embedding(audio, rgb, depth)
        dropped = self.final.dropout(embedding)
        polar_predict = self.final.polar_head_probe(dropped)
        angle_logits = self.final.angle_head_probe(dropped)
        action_logits = self.final.action_head(dropped)
        return polar_predict, angle_logits , action_logits

    def forward_joint(self, audio, rgb, depth):
        """
        Joint forward for action + angle training.
        - action_logits: from AV fusion branch
        - angle_logits: from audio encoder head (8 sectors)
        """
        _, _, action_logits = self.forward(audio, rgb, depth)
        if len(audio.shape) == 3:
            audio = audio.unsqueeze(0)
        angle_logits = self.audio_encoder(audio)
        return action_logits, angle_logits
    

    def embedding_forward(self, audio , rgb , depth=None):
        embedding = self._action_embedding(audio, rgb, depth)
        return  embedding

    def embedding_forward_modality(self, audio, rgb, depth, modality):
        """Return a frozen feature from one encoder branch, without fusion or masking."""
        if modality == "audio_only":
            if audio.ndim == 3:
                audio = audio.unsqueeze(0)
            encoded = self.audio_encoder.encoder_forward(audio)
            if encoded.ndim != 3 or encoded.shape[1] != 1:
                raise ValueError(f"Expected one audio token, got {tuple(encoded.shape)}")
            return encoded[:, 0, :]
        if modality == "vision_only":
            if rgb.ndim == 3:
                rgb = rgb.unsqueeze(0)
            rgbd = rgb.permute(0, 3, 1, 2)
            visual = spatial_tokens(self.visual_encoder(rgbd))
            visual = visual + self.add_position(visual)
            encoded = self.visual_transformer_encoder(visual)
            return encoded.mean(dim=1)
        raise ValueError(f"Unknown single-modality branch: {modality}")
    
    def embedding_forward_attention(self, audio , rgb , depth):
        if len(audio.shape) == 3:
            audio = audio.unsqueeze(0)
        if len(rgb.shape) == 3:
            rgb = rgb.unsqueeze(0)

        rgb = rgb.permute(0,3, 1, 2)

        rgbd = rgb
        
        
        audio_encoder = self.audio_encoder.encoder_forward(audio)
        visual_cnn = self.visual_encoder(rgbd)
        v_batch , v_dim , v_h , v_w = visual_cnn.shape
        visual_cnn = spatial_tokens(visual_cnn)
        v_position = self.add_position(visual_cnn)
        visual_cnn = visual_cnn + v_position
        visual_encoder = self.visual_transformer_encoder(visual_cnn)
        concat_encoder = torch.cat((audio_encoder , visual_encoder ) , dim=1)
        share_visual_audio_encoder = self.vaencoder(concat_encoder)
        p_share_encoder = self.add_position(share_visual_audio_encoder)
        share_encoder = share_visual_audio_encoder + p_share_encoder
        embedding = self.final.fc1(share_encoder)
        embedding = self.final.polar_head[0](embedding)
        embedding = embedding.squeeze(0)
        audio_encoder = audio_encoder.squeeze(0)
        return audio_encoder , embedding
        # return embedding

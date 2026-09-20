import sys
 
import torch
import torch.nn as nn
import torch.functional as F
from experiment.avnav.model.Encoder import Encoder

class PositionalEcoder(nn.Module):
    def __init__(self, max_len = 10000, d_models = 128 , device = None):
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = max_len
        self.d_models = d_models
        # 优先构造出整个位置编码矩阵
        self.register_buffer("position_matrix", torch.zeros(max_len, self.d_models, device=device), persistent=False)

        self.position_matrix.requires_grad = False # 1. 后续相加的时候生成的向量还会有梯度？后续相加后有梯度，但是位置编码不参与反向传播。

        # 位置编码矩阵参数
        self.pos = torch.arange(0 , self.max_len , step= 1).unsqueeze(1)
        self._2k = torch.arange(0 , self.d_models , step=2)

        # 填充位置编码矩阵
        self.position_matrix[:,0::2] = torch.sin(self.pos * (1000 ** (self._2k / self.d_models)))
        self.position_matrix[:,1::2] = torch.cos(self.pos * (1000 ** (self._2k / self.d_models)))
        
    def forward(self, x :torch.tensor) -> torch.tensor:
        batch,  len ,encode_dim ,= x.shape
        return self.position_matrix[:len,:]


# 这种方式向下提取，在这个任务上不一定能做的非常好
class VisualCNNEncoder(nn.Module):
    def __init__(self,   input_dim = 3, hidden_dim = 8, output_dim = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 图片shape （3 ， 256 ， 256）

        self.conv2d_1 = nn.Conv2d(in_channels=self.input_dim , out_channels= self.hidden_dim, kernel_size=2 , stride=2)
        self.conv2d_2 = nn.Conv2d(in_channels=self.hidden_dim , out_channels= 8* self.hidden_dim, kernel_size=4 , stride=4 , padding=1)
        self.conv2d_3 = nn.Conv2d(in_channels=8 * self.hidden_dim, out_channels=self.output_dim, kernel_size=4 , stride=4 , padding=1)

    def forward(self, visual):
        # pdb.set_trace()
        x_1 = self.conv2d_1(visual)
        # print(x_1.shape)# (batch , 8 , 64 , 64)
        x_2 = self.conv2d_2(x_1)
        # print(x_2.shape)# (batch , 64, 16 , 16)
        result = self.conv2d_3(x_2)
        # print(result.shape) # (batch , 128 . 4 ,4)
        # 原图像64 * 64 被编码成了一个patch ， 维度为128
        return result

# 这种方式向下提取，在这个任务上不一定能做的非常好
class AudioCNNEncoder(nn.Module):
    def __init__(self,   input_dim = 2, hidden_dim = 16, output_dim = 128 , sr=16000):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sr = sr

        # 必要的时候考虑通过线性层进行变换，然后让audio得到我们想要输入的形状

        self.conv2d_1 = nn.Conv2d(in_channels=self.input_dim , out_channels= self.hidden_dim, kernel_size=4 , stride=4 , padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_dim, kernel_size=4 , stride=4 , padding=1)

    def forward(self, mel_audio):
        # mel_audio = self._deal_audio(audio)
        
        # print(mel_audio.shape)
        
        x_1 = self.conv2d_1(mel_audio)
        # print(x_1.shape)# (batch , 16 , 32 , 16)
        result = self.conv2d_2(x_1)
        # print(result.shape)# (batch , 128, 8 , 4)
        return result


class VisualEncoder(nn.Module):
    def __init__(self,   input_dim = 8, kernel_size = 32, output_dim = 128):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.output_dim = output_dim

        # 图片shape （3 ， 256 ， 256）

        self.conv2d_1 = nn.Conv2d(in_channels=self.input_dim , out_channels= self.output_dim, kernel_size= self.kernel_size , stride=self.kernel_size)

    def forward(self, visual):
        # pdb.set_trace()
        x_1 = self.conv2d_1(visual)
        # print(x_1.shape)# (batch , 8 , 8)
        return x_1
    
class AudioEncoder(nn.Module):
    def __init__(self,   input_dim = 2, output_dim = 128):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv2d_1 = nn.Conv2d(in_channels=self.input_dim , out_channels= self.output_dim, kernel_size=16 , stride=16 , padding=1)

    def forward(self, mel_audio):
        # mel_audio = self._deal_audio(audio)
        
        # print(mel_audio.shape)
        
        x_1 = self.conv2d_1(mel_audio)
        # print(x_1.shape)# (batch , 16 , 32 , 16)
        return x_1
    

class AddPostional(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.positional = PositionalEcoder()
        self.visualencoder = VisualEncoder()
        self.audioencoder = AudioEncoder()
    
    def forward(self, audio ,visual):

        
        visual_encoder = self.visualencoder(visual)
        v_batch , v_dim , v_h , v_w = visual_encoder.shape
        audio_encoder = self.audioencoder(audio)
        a_batch , a_dim , a_h , a_w = audio_encoder.shape

        visual_encoder = visual_encoder.reshape(v_batch , v_h * v_w  , v_dim)
        audio_encoder = audio_encoder.reshape(a_batch ,   a_h * a_w  , a_dim)

        v_position = self.positional(visual_encoder)
        visual_encoder = visual_encoder + v_position

        a_position = self.positional(audio_encoder)
        audio_encoder = audio_encoder + a_position

        return audio_encoder , visual_encoder


class VAEncode(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_transformer_encoder = Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        # 根据音频的实际形状进行调整d_model参数
        self.audio_transformer_encoder = Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)
        self.share_visual_audio_encoder = Encoder(d_model=128 , ffn_hidden=64,n_head=4,n_layers=3,drop_prob=0.2)

    def forward(self ,audio , visual):
        visual_x = self.visual_transformer_encoder(visual)
        audio_x = self.audio_transformer_encoder(audio)
        # pdb.set_trace()
        avencode = torch.cat((audio_x , visual_x ) ,dim=1)
        combinencoder = self.share_visual_audio_encoder(avencode)
        # y = self.layer(combinencoder)
        return combinencoder
    
class VADecode(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio_visual_decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=128 , nhead=8 , batch_first=True) , num_layers= 6)
        self.binaural_audio_decoder = nn.TransformerDecoder(decoder_layer=nn.TransformerDecoderLayer(d_model=128 , nhead=8 , batch_first=True) , num_layers= 6)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    

    def forward(self, audio_visual_encoder, masked_binaural_audio):

        tgt_len = masked_binaural_audio.size(1)
        device = masked_binaural_audio.device
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)

        x = self.audio_visual_decoder(
            tgt=masked_binaural_audio,
            memory=audio_visual_encoder,
            tgt_mask=tgt_mask
        )

        y = self.binaural_audio_decoder(
            tgt=masked_binaural_audio,
            memory=x,
            tgt_mask=tgt_mask
        )

        return y

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.audioencode = AudioEncoder()
        self.positionnal = PositionalEcoder()
        self.addpostional = AddPostional()
        self.vaencode = VAEncode()
        self.vadecode = VADecode()
        
        self.hidden_dim = 1024

        self.down_dim_1 = nn.Linear(in_features=  20*128 , out_features= self.hidden_dim)
        self.down_dim_2 = nn.Linear(in_features= self.hidden_dim , out_features= self.hidden_dim // 2)
        self.activate = nn.ReLU()
    def forward(self , audio , visual):

        audio_encoder , visual_encoder = self.addpostional(audio , visual)
        
        y = self.vaencode(audio_encoder , visual_encoder)
        
        batch , y_token , y_d_model = y.shape
        y = y.reshape(batch , y_token * y_d_model) # y_token * y_d_modl = 32 * 128
        down_y1 = self.activate(self.down_dim_1(y))
        down_y2 = self.down_dim_2(down_y1)
        return down_y2

# models_transformer_acgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import math

# 既存のGeneratorを流用 (LSTMベースで安定しているため)
from models import Generator

# --- Positional Encoding (Transformerに順序情報を与える) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# --- Transformer Discriminator ---
class TransformerDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_heads=8, num_layers=2, max_len=1000, pretrained_embeddings=None):
        super(TransformerDiscriminator, self).__init__()
        
        # Embedding
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)
        
        # Transformer Encoder Block
        # batch_first=True にすることで (Batch, Seq, Dim) で扱える
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # AC-GANの2つの出力層 (Spectral Normで安定化)
        # Global Average Pooling後の次元は embedding_dim と同じ
        self.fc_validity = utils.spectral_norm(nn.Linear(embedding_dim, 1))
        self.fc_class = utils.spectral_norm(nn.Linear(embedding_dim, num_classes))

    def forward(self, sequence, soft_input=None):
        # 1. Embedding
        if soft_input is not None:
            # Generatorからの入力 (Softmax通過後)
            # soft_input: [Batch, Seq, Vocab]
            # embedding.weight: [Vocab, Dim]
            # -> [Batch, Seq, Dim]
            x = soft_input 
            # soft_inputの場合はPositional Encodingの適用が難しいが、
            # ここではシンプルに重み行列との積として扱う
        else:
            # Realデータ
            x = self.embedding(sequence) # [Batch, Seq, Dim]
        
        # 2. Scale & Positional Encoding
        x = x * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)
        
        # 3. Transformer Encoder
        # Padding Maskの作成 (0の部分を無視)
        if soft_input is None:
            src_key_padding_mask = (sequence == 0)
        else:
            src_key_padding_mask = None # FakeデータはPaddingがない前提(あるいは無視)
            
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 4. Global Average Pooling
        # パディング部分を考慮して平均を取る
        if src_key_padding_mask is not None:
            mask_expanded = src_key_padding_mask.unsqueeze(-1).expand(x.size()).float()
            x = x * (1.0 - mask_expanded)
            sum_x = x.sum(dim=1)
            count_x = (1.0 - mask_expanded).sum(dim=1).clamp(min=1)
            x_avg = sum_x / count_x
        else:
            x_avg = x.mean(dim=1)
            
        # 5. Output Heads
        validity = self.fc_validity(x_avg)
        class_logits = self.fc_class(x_avg)
        
        return validity, class_logits
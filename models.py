# models.py (BERT-AC-GAN Optuna対応版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
import math

# --- 1. ジェネレータ (変更なし) ---
class Generator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, noise_dim, num_classes, seq_length):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.label_embedding = nn.Embedding(num_classes, noise_dim)
        self.lstm = nn.LSTM(
            input_size=noise_dim * 2, 
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.unsqueeze(1).repeat(1, self.seq_length, 1)
        combined_input = torch.cat([noise, label_emb], dim=2)
        lstm_out, _ = self.lstm(combined_input)
        logits = self.fc(lstm_out)
        return logits

# --- 2. 位置エンコーディング (変更なし) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- 3. ディスクリミネータ (Optuna対応 Transformer) ---
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, nhead=8, num_layers=3, dropout=0.1):
        super(Discriminator, self).__init__()
        
        # Transformerの設定
        # hidden_dim を d_model として使う
        self.d_model = hidden_dim 
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer Encoder レイヤー (Optunaで探索するパラメータを適用)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=self.d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 出力層
        self.fc_validity = utils.spectral_norm(nn.Linear(self.d_model, 1))
        self.fc_class = utils.spectral_norm(nn.Linear(self.d_model, num_classes))

    def forward(self, sequence, soft_input=None):
        if soft_input is not None:
            # Generatorとの次元整合性のため、Embeddingと同じ次元であると仮定
            x = soft_input
        else:
            x = self.embedding(sequence)

        # スケーリングと位置エンコーディング
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Global Average Pooling
        x = x.mean(dim=1) 
        
        validity = self.fc_validity(x)
        class_logits = self.fc_class(x)
        
        return validity, class_logits
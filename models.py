# models.py (CNN-Discriminator 版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

# --- 1. ジェネレータ (LSTMのまま) ---
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

# --- 2. ディスクリミネータ (CNNに進化！) ---
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(Discriminator, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 1次元畳み込み層 (3種類のウィンドウサイズ)
        # フィルタ数(out_channels)を多めに設定
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        
        self.dropout = nn.Dropout(0.5)
        
        # 特徴量結合後の次元: 128 * 3 = 384
        feature_dim = 128 * 3
        
        # --- AC-GAN用の2つの出力層 ---
        # スペクトル正規化を入れて学習を安定させる
        self.fc_validity = utils.spectral_norm(nn.Linear(feature_dim, 1))
        self.fc_class = utils.spectral_norm(nn.Linear(feature_dim, num_classes))

    def forward(self, sequence, soft_input=None):
        # Generatorからのソフト入力に対応
        if soft_input is not None:
            x = soft_input
        else:
            x = self.embedding(sequence)
        
        # (Batch, SeqLen, EmbDim) -> (Batch, EmbDim, SeqLen)
        x = x.permute(0, 2, 1)
        
        # CNN + ReLU + MaxPool
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))
        
        # Global Max Pooling
        p1 = F.max_pool1d(c1, c1.shape[2]).squeeze(2)
        p2 = F.max_pool1d(c2, c2.shape[2]).squeeze(2)
        p3 = F.max_pool1d(c3, c3.shape[2]).squeeze(2)
        
        # 特徴ベクトル結合
        features = torch.cat([p1, p2, p3], dim=1)
        features = self.dropout(features)
        
        # 2つの判定結果を出力
        validity = self.fc_validity(features)
        class_logits = self.fc_class(features)
        
        return validity, class_logits
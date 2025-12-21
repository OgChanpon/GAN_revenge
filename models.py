# models.py (CNN + SE-Block + Full Spectral Norm 版)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils

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

# --- 2. SE-Block (変更なし) ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# --- 3. ディスクリミネータ (Spectral Norm 強化版) ---
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pretrained_embeddings=None):
        super(Discriminator, self).__init__()
        
        # ★ Embedding層の初期化分岐
        if pretrained_embeddings is not None:
            # 事前学習済み重みを使用 (freeze=False にして、GAN学習中に微調整させる)
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
            print(">> Discriminator: Pre-trained Embedding loaded.")
        else:
            # 従来通りランダム初期化
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        num_filters = 128
        
        # ★ 変更点: Conv1d層にも spectral_norm を適用 ★
        # これによりリプシッツ制約を満たしやすくなり、Dの学習が安定化します
        self.conv1 = utils.spectral_norm(nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=3))
        self.conv2 = utils.spectral_norm(nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=4))
        self.conv3 = utils.spectral_norm(nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=5))
        
        # SE-Block の追加
        self.se1 = SEBlock(num_filters)
        self.se2 = SEBlock(num_filters)
        self.se3 = SEBlock(num_filters)
        
        self.dropout = nn.Dropout(0.5)
        
        # 結合後の次元
        feature_dim = num_filters * 3
        
        # 出力層 (元々 spectral_norm が適用されていた箇所)
        self.fc_validity = utils.spectral_norm(nn.Linear(feature_dim, 1))
        self.fc_class = utils.spectral_norm(nn.Linear(feature_dim, num_classes))

    def forward(self, sequence, soft_input=None):
        if soft_input is not None:
            x = soft_input
        else:
            x = self.embedding(sequence)
        
        # (Batch, SeqLen, EmbDim) -> (Batch, EmbDim, SeqLen)
        x = x.permute(0, 2, 1)
        
        # CNN -> ReLU -> SE-Block -> MaxPool
        
        # Path 1
        c1 = F.relu(self.conv1(x))
        c1 = self.se1(c1)
        p1 = F.max_pool1d(c1, c1.shape[2]).squeeze(2)
        
        # Path 2
        c2 = F.relu(self.conv2(x))
        c2 = self.se2(c2)
        p2 = F.max_pool1d(c2, c2.shape[2]).squeeze(2)
        
        # Path 3
        c3 = F.relu(self.conv3(x))
        c3 = self.se3(c3)
        p3 = F.max_pool1d(c3, c3.shape[2]).squeeze(2)
        
        # 結合
        features = torch.cat([p1, p2, p3], dim=1)
        features = self.dropout(features)
        
        validity = self.fc_validity(features)
        class_logits = self.fc_class(features)
        
        return validity, class_logits
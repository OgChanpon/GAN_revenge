# models_lstm_acgan.py
import torch
import torch.nn as nn

# Generator (提案手法のLSTM版Generatorと同じ)
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, noise_dim, num_classes, seq_length):
        super(LSTMGenerator, self).__init__()
        self.seq_length = seq_length
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

# Discriminator (LSTM版)
class LSTMDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # 双方向ではなく通常のLSTM (Optuna指定がないため一般的構成) または 双方向
        # ここでは前の実験と条件を揃えるため Bidirectional=True 推奨だが
        # AC-GANのDとしては通常のLSTMもよく使われる。ここでは汎用性重視で Bidirectional=True にします。
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        
        # 特徴量次元: hidden_dim * 2 (双方向分)
        feature_dim = hidden_dim * 2
        
        # 出力層 (真贋 + クラス)
        self.fc_validity = nn.Linear(feature_dim, 1)
        self.fc_class = nn.Linear(feature_dim, num_classes)

    def forward(self, sequence, soft_input=None):
        if soft_input is not None:
            x = soft_input
        else:
            x = self.embedding(sequence)
        
        out, _ = self.lstm(x)
        # 最後のタイムステップ
        # Global Max Poolingの方が安定するが、Optuna設定時の構造に従うべき。
        # 不明な場合は Max Pooling が無難。
        feature = torch.max(out, dim=1)[0]
        
        validity = self.fc_validity(feature)
        class_logits = self.fc_class(feature)
        return validity, class_logits
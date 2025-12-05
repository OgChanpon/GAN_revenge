# models.py (AC-GAN用)

import torch
import torch.nn as nn
import torch.nn.utils as utils

# --- 1. ジェネレータ (偽造職人) ---
class Generator(nn.Module):
    def __init__(self, vocab_size, hidden_dim, noise_dim, num_classes, seq_length):
        super(Generator, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        
        # クラスラベルを埋め込む層 (ラベル -> ベクトル)
        self.label_embedding = nn.Embedding(num_classes, noise_dim)
        
        # LSTMへの入力層 (ノイズ + ラベル情報)
        # ノイズとラベルを足し合わせるか結合するかは手法によるが、ここでは結合して入力次元を2倍にする
        self.lstm = nn.LSTM(
            input_size=noise_dim * 2, 
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # 出力層 (各タイムステップで次のAPIを予測)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise, labels):
        # noise: (Batch, SeqLen, NoiseDim) - 実際にはSeqLen=1で入力してループさせるのが一般的だが、
        # ここでは簡易化のため、SeqLen分のノイズを一気に入力する方式をとる
        
        # labels: (Batch) -> (Batch, NoiseDim)
        label_emb = self.label_embedding(labels)
        
        # シーケンス長に合わせてラベル情報を複製
        # (Batch, NoiseDim) -> (Batch, SeqLen, NoiseDim)
        label_emb = label_emb.unsqueeze(1).repeat(1, self.seq_length, 1)
        
        # ノイズとラベルを結合
        # (Batch, SeqLen, NoiseDim * 2)
        combined_input = torch.cat([noise, label_emb], dim=2)
        
        lstm_out, _ = self.lstm(combined_input)
        
        # ロジット(確率の元)を出力
        # (Batch, SeqLen, VocabSize)
        logits = self.fc(lstm_out)
        return logits

# --- 2. ディスクリミネータ (鑑定士 + 分類器) ---
class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(Discriminator, self).__init__()
        
        # スペクトル正規化を使用 (学習安定化のため)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 最終層の特徴量サイズ (Bidirectionalなので2倍)
        feature_dim = hidden_dim * 2
        
        # --- 出力層 A: 本物か偽物か (Validity) ---
        self.fc_validity = utils.spectral_norm(nn.Linear(feature_dim, 1))
        
        # --- 出力層 B: どのクラスか (Class Classification) ---
        self.fc_class = utils.spectral_norm(nn.Linear(feature_dim, num_classes))

    def forward(self, sequence, soft_input=None):
        # GANの学習のために、ソフトな入力(Generator由来)とハードな入力(本物由来)の両方に対応
        if soft_input is not None:
            embedded = soft_input
        else:
            embedded = self.embedding(sequence)
        
        # LSTM処理
        lstm_out, _ = self.lstm(embedded)
        
        # 最後のタイムステップの隠れ状態を取得
        last_hidden_state = lstm_out[:, -1, :]
        
        # 2つの判定を行う
        validity = self.fc_validity(last_hidden_state) # Real(1) or Fake(0)
        class_logits = self.fc_class(last_hidden_state) # Class 0~7
        
        return validity, class_logits
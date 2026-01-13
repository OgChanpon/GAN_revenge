# models_lstm.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # Bidirectional=True で文脈を考慮
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # hidden_dim * 2 (双方向) -> num_classes
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        # lstm_out: (batch, seq, hidden*2)
        lstm_out, _ = self.lstm(x)
        # Global Max Pooling (時系列の中で最も強い特徴を取り出す)
        # 単純な最後のステップ取得より精度が出やすい
        x = torch.max(lstm_out, dim=1)[0]
        logits = self.fc(x)
        return logits
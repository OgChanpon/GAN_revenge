# evaluate_transformer.py
import torch
import torch.nn as nn
import pickle
import numpy as np
import math
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 設定 ---
MODEL_PATH = 'transformer_best.pth'
TEST_DATA_PATH = 'test_dataset.pkl'
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Class Definition (Must match training script) ---
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

class MalwareTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_heads, num_layers, max_len, pretrained_embeddings=None):
        super(MalwareTransformer, self).__init__()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(0.0)

    def forward(self, src):
        mask = (src == 0)
        x = self.embedding(src) * math.sqrt(256)
        x = self.pos_encoder(x)
        # 評価時はDropoutなし
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        mask_expanded = mask.unsqueeze(-1).expand(output.size()).float()
        output = output * (1.0 - mask_expanded)
        sum_output = output.sum(dim=1)
        count_output = (1.0 - mask_expanded).sum(dim=1).clamp(min=1)
        avg_output = sum_output / count_output
        logits = self.fc(avg_output)
        return logits

def evaluate():
    print(f"Loading data and model...")
    with open(WORD_TO_INT_PATH, 'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)

    with open(CATEGORY_TO_ID_PATH, 'rb') as f:
        cat_to_id = pickle.load(f)
    NUM_CLASSES = len(cat_to_id)
    
    # Load Model
    model = MalwareTransformer(VOCAB_SIZE, 256, NUM_CLASSES, 8, 2, 1000).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load Test Data
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f)

    y_true = []
    y_pred = []

    print("Evaluating...")
    with torch.no_grad():
        for seq, label in test_data:
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(seq_tensor)
            pred_label = torch.argmax(logits, dim=1).item()
            y_true.append(label)
            y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Transformer Evaluation Result ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    target_names = [id_to_cat[i] for i in range(NUM_CLASSES)]
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate()
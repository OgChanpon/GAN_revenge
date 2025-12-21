# train_transformer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import math
import os
import copy
from collections import Counter
from data_preprocessing import prepare_data_loaders

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 0.0001
WEIGHT_DECAY = 1e-4

EMBEDDING_DIM = 256
NUM_HEADS = 8       # Attention Headの数
NUM_LAYERS = 2      # Transformer層の数
DROPOUT = 0.3
MAX_SEQUENCE_LENGTH = 1000

SAVE_PATH = 'transformer_best.pth'
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id.pkl'

# --- Transformer Model Definition ---
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
        
        # Embedding
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False, padding_idx=0)
            print(">> Pre-trained Embedding loaded.")
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, dropout=DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, src):
        # src: [Batch, Seq_Len]
        mask = (src == 0) # Padding mask
        
        x = self.embedding(src) * math.sqrt(EMBEDDING_DIM)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transformer Pass
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global Average Pooling (シーケンス全体の平均を取る)
        # パディング部分(0)の影響を消すための処理
        mask_expanded = mask.unsqueeze(-1).expand(output.size()).float()
        output = output * (1.0 - mask_expanded) # マスク部分は0にする
        
        sum_output = output.sum(dim=1)
        count_output = (1.0 - mask_expanded).sum(dim=1).clamp(min=1) # 0除算防止
        
        avg_output = sum_output / count_output
        
        logits = self.fc(avg_output)
        return logits

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# --- Main Training Logic ---
def train():
    # 1. Load Data
    if not os.path.exists(WORD_TO_INT_PATH):
        print("Error: 辞書ファイルが見つかりません")
        return

    with open(WORD_TO_INT_PATH, "rb") as f:
        word_to_int = pickle.load(f)
    VOCAB_SIZE = len(word_to_int)

    with open(CATEGORY_TO_ID_PATH, "rb") as f:
        category_to_id = pickle.load(f)
    NUM_CLASSES = len(category_to_id)
    
    print("データをロード中...")
    train_loader, val_loader = prepare_data_loaders(
        batch_size=BATCH_SIZE, 
        max_length=MAX_SEQUENCE_LENGTH,
        validation_split=0.1, 
        balance_data=False
    )

    # 2. Calculate Class Weights
    all_labels = train_loader.dataset.tensors[1].cpu().numpy()
    label_counts = Counter(all_labels)
    total_samples = len(all_labels)
    class_weights = []
    for i in range(NUM_CLASSES):
        count = label_counts.get(i, 0)
        weight = total_samples / (count if count > 0 else 1)
        class_weights.append(weight)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

    # 3. Load Word2Vec Weights (if available)
    embedding_weights = None
    if os.path.exists("embedding_matrix.pth"):
        embedding_weights = torch.load("embedding_matrix.pth").to(device)

    # 4. Initialize Model
    model = MalwareTransformer(
        VOCAB_SIZE, EMBEDDING_DIM, NUM_CLASSES, NUM_HEADS, NUM_LAYERS, MAX_SEQUENCE_LENGTH, embedding_weights
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

    # 5. Training Loop
    print("\n--- Transformer Training Start ---")
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience = 8
    counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                logits = model(seqs)
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
            print(f" -> Best Val Acc! (Saved memory)")
        else:
            counter += 1
            if counter >= patience:
                print("Early Stopping.")
                break

    # Save
    print(f"Saving best model to {SAVE_PATH}...")
    torch.save(best_model_wts, SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    train()
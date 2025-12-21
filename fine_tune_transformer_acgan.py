# fine_tune_transformer_acgan.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import copy
from collections import Counter
from data_preprocessing import prepare_data_loaders

# ★ Transformerモデル定義を読み込み
from models_transformer_acgan import TransformerDiscriminator

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
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ★ Optuna Best Params を適用 ★
BATCH_SIZE = 32
LR = 5.177e-05
GAMMA = 4.729
WEIGHT_DECAY = 5.908e-05 
NUM_EPOCHS = 50   

# パス設定
BASE_MODEL_PATH = 'transformer_acgan_discriminator.pth' # AC-GANで学習したモデル
SAVE_PATH = 'transformer_acgan_finetuned.pth'         # 完成形

WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl'

# モデル設定 (学習時と合わせる)
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 1000
NUM_HEADS = 8
NUM_LAYERS = 2

# --- データ準備 ---
if not os.path.exists(CATEGORY_TO_ID_PATH):
    print("エラー: フィルタ済み辞書が見つかりません。")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f: vocab_size = len(pickle.load(f))
with open(CATEGORY_TO_ID_PATH, "rb") as f: num_classes = len(pickle.load(f))

print("データをロード中...")
train_loader, val_loader = prepare_data_loaders(
    batch_size=BATCH_SIZE, max_length=MAX_LEN, validation_split=0.1, balance_data=False
)

# クラス重み
all_labels = train_loader.dataset.tensors[1].cpu().numpy()
counts = Counter(all_labels)
weights = [len(all_labels) / (counts[i] if counts[i]>0 else 1) for i in range(num_classes)]
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
class_weights_tensor /= class_weights_tensor.mean()

# --- モデル準備 ---
print(f"ベースモデル {BASE_MODEL_PATH} をロード中...")
model = TransformerDiscriminator(
    vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes, 
    num_heads=NUM_HEADS, num_layers=NUM_LAYERS, max_len=MAX_LEN
).to(device)

try:
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    print("ロード成功。Fine-tuningを開始します。")
except Exception as e:
    print(f"ロードエラー: {e}")
    exit()

# Optimizer & Loss
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=GAMMA)

# --- Fine-tuning Loop ---
print("\n--- Transformer Discriminator Fine-tuning ---")

best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience = 10 
counter = 0

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0
    
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        # AC-GANのDは (validity, class_logits) を返すので [1] だけ使う
        _, class_logits = model(seqs) 
        loss = criterion(class_logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(class_logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            _, class_logits = model(seqs)
            _, predicted = torch.max(class_logits.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
    
    # Early Stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        counter = 0 
        print(f"  -> Best Validation Accuracy Updated: {best_val_acc:.2f}% (Saved)")
    else:
        counter += 1
        print(f"  -> No improvement. Patience: {counter}/{patience}")
        
    if counter >= patience:
        print("Early Stopping.")
        break

# 保存
print(f"\nベストモデル保存: {SAVE_PATH} (Val Acc: {best_val_acc:.2f}%)")
torch.save(best_model_wts, SAVE_PATH)
print("完了")
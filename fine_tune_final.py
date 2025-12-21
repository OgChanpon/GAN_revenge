# fine_tune_final.py (Optuna Optimized & Fixed)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import copy
from collections import Counter

# 既存モジュール
from models import Discriminator
from data_preprocessing import prepare_data_loaders

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

# --- 設定 (Optunaのベストパラメータ) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ★★★ Optuna Best Params ★★★
BATCH_SIZE = 32
LR = 5.177e-05        # 約 0.00005
GAMMA = 4.729         # Focal Lossの強度
WEIGHT_DECAY = 5.908e-05 

NUM_EPOCHS = 50   

# パス設定
BASE_MODEL_PATH = 'acgan_discriminator_filtered.pth' # フィルタ後のモデル
SAVE_PATH = 'acgan_final_best.pth'                   # 保存名

WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl'

# --- データの準備 ---
if not os.path.exists(WORD_TO_INT_PATH):
    print("エラー: 辞書ファイルがありません")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open(CATEGORY_TO_ID_PATH, "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)
print(f"クラス数: {NUM_CLASSES}")

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_SEQUENCE_LENGTH = 1000

print("データをロード中 (Train/Val分割)...")
train_loader, val_loader = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.1, 
    balance_data=False
)

# --- ★ 修正箇所: クラス重みの計算 ★ ---
# TensorDatasetから直接テンソルを取得
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

# --- モデルの準備 ---
print(f"ベースモデル {BASE_MODEL_PATH} をロード中...")
discriminator = Discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

try:
    discriminator.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    print("ロード成功。")
except Exception as e:
    print(f"ロードエラー: {e}")
    # サイズ不一致時の救済措置
    print("重みの一部だけロードを試みます...")
    pretrained_dict = torch.load(BASE_MODEL_PATH, map_location=device)
    model_dict = discriminator.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    discriminator.load_state_dict(model_dict)
    print("部分ロード完了。")

# Optimizer & Loss (Optunaパラメータ適用)
optimizer = optim.Adam(discriminator.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=GAMMA)

# --- 学習ループ ---
print("\n--- Final Fine-tuning Start ---")

best_val_acc = 0.0
best_model_wts = copy.deepcopy(discriminator.state_dict())
patience = 10 
counter = 0

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    discriminator.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        _, class_logits = discriminator(seqs)
        loss = criterion(class_logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(class_logits.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # --- Validation Phase ---
    discriminator.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for seqs, labels in val_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            _, class_logits = discriminator(seqs)
            loss = criterion(class_logits, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(class_logits.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    avg_val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%")
    
    # --- Early Stopping Logic ---
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(discriminator.state_dict())
        counter = 0 
        print(f"  -> Best Validation Accuracy Updated: {best_val_acc:.2f}% (Saved)")
    else:
        counter += 1
        print(f"  -> No improvement. Patience: {counter}/{patience}")
        
    if counter >= patience:
        print("Early Stopping.")
        break

# --- 保存 ---
print(f"\nベストモデル保存: {SAVE_PATH} (Val Acc: {best_val_acc:.2f}%)")
torch.save(best_model_wts, SAVE_PATH)
print("完了")
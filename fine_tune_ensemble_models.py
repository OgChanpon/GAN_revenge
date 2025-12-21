# fine_tune_ensemble_models.py (修正版)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import copy
from collections import Counter
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
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ★ Optuna Best Params ★
BATCH_SIZE = 32
LR = 5.177e-05
GAMMA = 4.729
WEIGHT_DECAY = 5.908e-05 
NUM_EPOCHS = 30  
NUM_MODELS = 5

WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl'

# --- データ準備 ---
if not os.path.exists(WORD_TO_INT_PATH):
    print("エラー: 辞書ファイルが見つかりません。")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f: vocab_size = len(pickle.load(f))
with open(CATEGORY_TO_ID_PATH, "rb") as f: num_classes = len(pickle.load(f))

print("データをロード中...")
train_loader, val_loader = prepare_data_loaders(
    batch_size=BATCH_SIZE, max_length=1000, validation_split=0.1, balance_data=False
)

# --- ★ 修正箇所: クラス重みの計算 ★ ---
# TensorDatasetから直接ラベルを取得
all_labels = train_loader.dataset.tensors[1].cpu().numpy()

counts = Counter(all_labels)
weights = [len(all_labels) / (counts[i] if counts[i]>0 else 1) for i in range(num_classes)]
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
class_weights_tensor /= class_weights_tensor.mean()

# --- ループ処理 ---
print(f"\n=== Fine-tuning 5 Ensemble Models ===")

for i in range(NUM_MODELS):
    load_path = f"acgan_ensemble_discriminator_{i}.pth"
    save_path = f"finetuned_ensemble_{i}.pth"
    
    if not os.path.exists(load_path):
        print(f"Skipping {load_path} (Not found)")
        continue

    print(f"\n>> Model {i+1}/{NUM_MODELS}: Loading {load_path}...")
    
    # モデル初期化 & ロード
    model = Discriminator(vocab_size, 256, 512, num_classes).to(device)
    model.load_state_dict(torch.load(load_path, map_location=device))
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=GAMMA)
    
    # Fine-tuning Loop
    best_val_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience = 8
    counter = 0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seqs)[1], labels) # class_logitsのみ使用
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                _, predicted = torch.max(model(seqs)[1].data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        
        # Early Stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    print(f"   Finished. Best Val Acc: {best_val_acc:.4f}")
    torch.save(best_wts, save_path)
    print(f"   Saved to {save_path}")

print("\n全モデルのFine-tuning完了！")
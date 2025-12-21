# optimize_with_optuna.py
import optuna
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

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = 'acgan_discriminator_filtered.pth' # Step 3で作ったモデル
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl' # フィルタ済み辞書
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_TRIALS = 100  # 試行回数 (多いほど良いが見つかるまで時間がかかる)

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

# --- データ準備 (Optunaの中で何度も呼ばないように外で一度だけ計算) ---
if not os.path.exists(CATEGORY_TO_ID_PATH):
    print("エラー: ファイルが見つかりません")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open(CATEGORY_TO_ID_PATH, "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)

print("データをロード中...")
# ここでは一旦バッチサイズ64でロードして、datasetオブジェクトだけ取得する
temp_loader, val_loader_fixed = prepare_data_loaders(
    batch_size=64, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.1, 
    balance_data=False
)
train_dataset = temp_loader.dataset
val_dataset = val_loader_fixed.dataset

# クラス重みの計算
all_labels = train_dataset.tensors[1].cpu().numpy()
label_counts = Counter(all_labels)
total_samples = len(all_labels)
class_weights = []
for i in range(NUM_CLASSES):
    count = label_counts.get(i, 0)
    weight = total_samples / (count if count > 0 else 1)
    class_weights.append(weight)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

# --- Optuna Objective Function ---
def objective(trial):
    # 1. 探索するハイパーパラメータの定義
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.5, 5.0) # Focal Lossの強度
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    
    # 2. データローダーの再構築 (バッチサイズが変わるため)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. モデルの初期化とロード
    model = Discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
    try:
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
    except Exception:
        # 重みサイズ不一致回避 (念のため)
        pretrained_dict = torch.load(BASE_MODEL_PATH, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=gamma)
    
    # 5. Training Loop (高速化のためEpoch数は少なめに設定)
    n_epochs = 15
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        model.train()
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            _, class_logits = model(seqs)
            loss = criterion(class_logits, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(device), labels.to(device)
                _, class_logits = model(seqs)
                _, predicted = torch.max(class_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        # Pruning (見込みのない試行を早期打ち切り)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_val_acc

if __name__ == "__main__":
    print("Optunaによる探索を開始します...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=NUM_TRIALS)

    print("\n==================================")
    print("Best Parameters Found:")
    print(study.best_params)
    print(f"Best Val Accuracy: {study.best_value:.4f}")
    print("==================================")
    
    # 結果をテキストに保存
    with open("optuna_results.txt", "w") as f:
        f.write(str(study.best_params))
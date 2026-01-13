# train_cnn_only.py
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import copy
from models import Discriminator
from data_preprocessing import prepare_data_loaders

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 50  # 比較用なので50で十分（提案手法に合わせるなら増やしてもOK）
LR = 0.001
SAVE_PATH = "cnn_only_best.pth"

# 辞書ファイル等のパス
WORD_TO_INT = 'word_to_int.pkl'
CAT_TO_ID = 'category_to_id_filtered_full.pkl'

def main():
    print(f"=== 実験1: 1D-CNN (GANなし) 学習開始 ===")
    
    # 1. データ準備
    if not os.path.exists(WORD_TO_INT):
        print("エラー: 辞書ファイルがありません。tokenka.pyを実行してください。")
        return
        
    with open(WORD_TO_INT, 'rb') as f: vocab_size = len(pickle.load(f))
    with open(CAT_TO_ID, 'rb') as f: num_classes = len(pickle.load(f))

    # DataLoader (data_preprocessing.pyを利用)
    train_loader, val_loader = prepare_data_loaders(
        batch_size=BATCH_SIZE, 
        max_length=1000, 
        validation_split=0.1, 
        balance_data=False # 不均衡のまま
    )

    # 2. モデル定義 (提案手法のDiscriminatorを流用)
    model = Discriminator(vocab_size, 256, 512, num_classes).to(DEVICE)
    
    # 3. 学習設定
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss() # シンプルなCE Loss

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    # 4. 学習ループ
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # Discriminatorの戻り値は (validity, class_logits)
            _, logits = model(seqs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(logits, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
        train_acc = correct / total
        
        # Validation
        val_acc = 0
        if val_loader:
            model.eval()
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for v_seqs, v_labels in val_loader:
                    v_seqs, v_labels = v_seqs.to(DEVICE), v_labels.to(DEVICE)
                    _, v_logits = model(v_seqs)
                    _, v_pred = torch.max(v_logits, 1)
                    v_total += v_labels.size(0)
                    v_correct += (v_pred == v_labels).sum().item()
            val_acc = v_correct / v_total
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # 保存
    torch.save(best_weights, SAVE_PATH)
    print(f"学習完了。ベストモデルを保存しました: {SAVE_PATH} (Val Acc: {best_val_acc:.4f})")

if __name__ == '__main__':
    main()
# train_classifier.py (純粋な分類器による比較実験)

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score

# 同じモデル構造を使って公平に比較する
from models import Discriminator
from data_preprocessing import prepare_data_loaders

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

BATCH_SIZE = 64
NUM_EPOCHS = 100 # GANより収束が早いので少なめでOK
LR = 0.0002      # 一般的な分類器の学習率

# 辞書ロード
with open("word_to_int.pkl", "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open("category_to_id.pkl", "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
MAX_SEQUENCE_LENGTH = 1000 

# --- データローダー ---
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0
)

# --- モデル準備 ---
# AC-GANのDiscriminatorと同じクラスを使用
model = Discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

# --- 損失関数とオプティマイザ ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- 学習ループ ---
print(f"\n--- 純粋な分類器の学習開始 ---")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (seqs, labels) in enumerate(train_loader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Discriminatorは (validity, class_logits) を返すが、
        # 今回は class_logits だけを使う
        _, outputs = model(seqs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {running_loss/len(train_loader):.4f} Acc: {epoch_acc:.2f}%")

# --- 保存 ---
print("\n--- 学習完了 ---")
torch.save(model.state_dict(), "classifier_only.pth")
print("モデル保存完了: classifier_only.pth")

# --- 簡易評価 (テストデータ) ---
print("\n--- テストデータでの評価 ---")
with open("test_dataset.pkl", "rb") as f:
    test_data = pickle.load(f)

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for seq, label in test_data:
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
        _, outputs = model(seq_tensor) # class_logitsのみ使用
        pred = torch.argmax(outputs, dim=1).item()
        y_true.append(label)
        y_pred.append(pred)

final_acc = accuracy_score(y_true, y_pred)
print(f"最終テスト精度: {final_acc:.4f} ({final_acc*100:.2f}%)")
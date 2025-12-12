import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
import torch.nn.functional as F

# 必要なモジュール
from models import Generator, Discriminator
from data_preprocessing import prepare_data_loaders

# --- ★ Focal Loss クラスの定義 ★ ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 標準的なCrossEntropyを計算 (reductionなしで個別に取得)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 確率 pt を計算 (exp(-loss))
        pt = torch.exp(-ce_loss)
        
        # Focal Lossの公式: (1 - pt)^gamma * log(pt)
        # 簡単なサンプル(ptが高い)ほど (1-pt) が0に近づき、損失が小さくなる
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 1. 設定とハイパーパラメータ (Optuna Best) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# 最強設定
EMBEDDING_DIM = 1024
HIDDEN_DIM = 512
NOISE_DIM = 100

LR_D = 0.000115
LR_G = 0.00013

K_STEPS = 2
G_STEPS = 2

LABEL_REAL = 0.85
LABEL_FAKE = 0.20

BATCH_SIZE = 64
NUM_EPOCHS = 100
MAX_SEQUENCE_LENGTH = 1000 

# --- 2. データの準備 ---
if not os.path.exists("word_to_int.pkl") or not os.path.exists("category_to_id.pkl"):
    print("エラー: 辞書ファイルが見つかりません。")
    exit()

with open("word_to_int.pkl", "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open("category_to_id.pkl", "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)
print(f"クラス数: {NUM_CLASSES}")

train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0,
    balance_data=True
)

# --- 3. モデルの初期化 ---
generator = Generator(VOCAB_SIZE, HIDDEN_DIM, NOISE_DIM, NUM_CLASSES, MAX_SEQUENCE_LENGTH).to(device)
discriminator = Discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

# --- 4. 最適化アルゴリズムと損失関数 ---
optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

adversarial_loss = nn.BCEWithLogitsLoss()

# ★ 損失関数を変更: CrossEntropy -> FocalLoss
print("Loss Function: Focal Loss (gamma=2.0) を使用します")
#auxiliary_loss = FocalLoss(gamma=2.0).to(device)
print("Loss Function: CrossEntropyLoss (Standard) を使用します")
auxiliary_loss = nn.CrossEntropyLoss()

# --- 5. 学習ループ ---
print(f"\n--- AC-GAN 最終決戦 (Focal Loss ver.) 開始 ---")

for epoch in range(NUM_EPOCHS):
    for i, (real_seqs, real_labels) in enumerate(train_loader):
        
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(device)
        real_labels = real_labels.to(device)

        valid = torch.full((batch_size, 1), LABEL_REAL, device=device)
        fake = torch.full((batch_size, 1), LABEL_FAKE, device=device)

        # ===============================================
        #  Train Discriminator
        # ===============================================
        optimizer_d.zero_grad()

        # Real
        pred_validity, pred_class = discriminator(real_seqs)
        d_loss_real_val = adversarial_loss(pred_validity, valid)
        d_loss_real_class = auxiliary_loss(pred_class, real_labels) # Focal Loss
        d_loss_real = d_loss_real_val + d_loss_real_class

        # Fake
        z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        
        fake_logits = generator(z, gen_labels)
        fake_seqs_discrete = torch.argmax(fake_logits, dim=2)
        
        pred_validity_fake, pred_class_fake = discriminator(fake_seqs_discrete.detach())
        d_loss_fake_val = adversarial_loss(pred_validity_fake, fake)
        d_loss_fake = d_loss_fake_val 

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # ===============================================
        #  Train Generator
        # ===============================================
        g_loss_val = 0.0

        if i % K_STEPS == 0:
            for _ in range(G_STEPS):
                optimizer_g.zero_grad()

                z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
                gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
                
                fake_logits = generator(z, gen_labels)
                fake_probs = F.softmax(fake_logits, dim=2)
                embed_matrix = discriminator.embedding.weight
                soft_input = torch.matmul(fake_probs, embed_matrix)
                
                pred_validity, pred_class = discriminator(None, soft_input=soft_input)

                valid_target = torch.full((batch_size, 1), 1.0, device=device)
                
                g_loss_validity = adversarial_loss(pred_validity, valid_target)
                g_loss_class = auxiliary_loss(pred_class, gen_labels) # Focal Loss
                
                g_loss = g_loss_validity + g_loss_class
                g_loss.backward()
                optimizer_g.step()
                
                g_loss_val = g_loss.item()

        if i % 50 == 0:
            print(
                f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss_val:.4f}]"
            )

    if (epoch + 1) % 10 == 0:
        print(f"--- Epoch {epoch+1} 完了 (モデル保存) ---")
        torch.save(generator.state_dict(), "acgan_generator_focal.pth")
        torch.save(discriminator.state_dict(), "acgan_discriminator_focal.pth")

print("\n--- 全学習工程完了 ---")
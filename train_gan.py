# train_gan.py (両側ラベルスムージング対応版)

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
import torch.nn.functional as F

# --- 必要なモジュールをインポート ---
from models import Generator, Discriminator
from data_preprocessing import prepare_data_loaders

# --- 1. 設定とハイパーパラメータ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ★★★ チューニング設定 ★★★
# ---------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 50 

# 学習率 (TTUR)
LR_D = 0.0002 
LR_G = 0.0002 

# K-step (Dの更新回数)
K_STEPS = 3

# ★ 両側ラベルスムージング (Two-sided Label Smoothing) ★
# Dの過信を防ぐため、0.0 / 1.0 ではなく、少し緩めた値を使う
LABEL_REAL = 0.9  # 本物の正解ラベル (通常 1.0 -> 0.9)
LABEL_FAKE = 0.1  # 偽物の正解ラベル (通常 0.0 -> 0.1)
# ---------------------------------------------

# 辞書とデータのロード
if not os.path.exists("word_to_int.pkl") or not os.path.exists("category_to_id.pkl"):
    print("エラー: 辞書ファイルが見つかりません。tokenka.py を実行してください。")
    exit()

with open("word_to_int.pkl", "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open("category_to_id.pkl", "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)
print(f"クラス数: {NUM_CLASSES}")

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NOISE_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 

# --- 2. データローダーの準備 ---
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0
)

# --- 3. モデルの初期化 ---
generator = Generator(VOCAB_SIZE, HIDDEN_DIM, NOISE_DIM, NUM_CLASSES, MAX_SEQUENCE_LENGTH).to(device)
discriminator = Discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

# --- 4. 最適化アルゴリズムと損失関数 ---
optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

adversarial_loss = nn.BCEWithLogitsLoss() # 本物か偽物か
auxiliary_loss = nn.CrossEntropyLoss()    # どのクラスか

# --- 5. 学習ループ ---
print(f"\n--- AC-GAN 学習開始 ---")
print(f"設定: K_STEPS={K_STEPS}, Real={LABEL_REAL}, Fake={LABEL_FAKE}")

for epoch in range(NUM_EPOCHS):
    for i, (real_seqs, real_labels) in enumerate(train_loader):
        
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(device)
        real_labels = real_labels.to(device)

        # ★ ラベル定義（両側スムージング適用）★
        valid = torch.full((batch_size, 1), LABEL_REAL, device=device) # 例: 0.9
        fake = torch.full((batch_size, 1), LABEL_FAKE, device=device)  # 例: 0.1

        # ===============================================
        #  Train Discriminator
        # ===============================================
        optimizer_d.zero_grad()

        # 1. 本物のデータを判定
        pred_validity, pred_class = discriminator(real_seqs)
        d_loss_real_val = adversarial_loss(pred_validity, valid)
        d_loss_real_class = auxiliary_loss(pred_class, real_labels)
        d_loss_real = d_loss_real_val + d_loss_real_class

        # 2. 偽物のデータを判定
        z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        
        # D学習用なので離散化(argmax)する
        fake_logits = generator(z, gen_labels)
        fake_seqs_discrete = torch.argmax(fake_logits, dim=2)
        
        pred_validity_fake, pred_class_fake = discriminator(fake_seqs_discrete.detach())
        
        # 偽物に対する損失
        d_loss_fake_val = adversarial_loss(pred_validity_fake, fake)
        d_loss_fake = d_loss_fake_val 

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # ===============================================
        #  Train Generator (K_STEPS回に1回)
        # ===============================================
        g_loss_val = 0.0

        if i % K_STEPS == 0:
            optimizer_g.zero_grad()

            # 再生成 (勾配用)
            z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
            gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
            fake_logits = generator(z, gen_labels)
            
            # ソフトな埋め込み
            fake_probs = F.softmax(fake_logits, dim=2)
            embed_matrix = discriminator.embedding.weight
            soft_input = torch.matmul(fake_probs, embed_matrix)
            
            pred_validity, pred_class = discriminator(None, soft_input=soft_input)

            # Gの目標: Dを完全に騙したいので、ここではスムージングせず「1.0」を目指すのが一般的
            # (ただし LABEL_REAL を目指す設定もあり得るが、強い勾配を得るため 1.0 推奨)
            valid_target = torch.full((batch_size, 1), 1.0, device=device)
            
            g_loss_validity = adversarial_loss(pred_validity, valid_target)
            g_loss_class = auxiliary_loss(pred_class, gen_labels)
            
            g_loss = g_loss_validity + g_loss_class
            g_loss.backward()
            optimizer_g.step()
            
            g_loss_val = g_loss.item()

        if i % 100 == 0:
            print(
                f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss_val:.4f}]"
            )

    # --- 生成サンプル確認 ---
    print(f"--- Epoch {epoch+1} ---")
    
# --- 保存 ---
print("\n--- 学習完了。モデルを保存します ---")
torch.save(generator.state_dict(), "acgan_generator.pth")
torch.save(discriminator.state_dict(), "acgan_discriminator.pth")
print("保存完了")
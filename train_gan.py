# train_gan.py (ダウンサンプリング廃止 + Focal Loss 対応版)

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
import torch.nn.functional as F
from collections import Counter

# --- 必要なモジュールをインポート ---
from models import Generator, Discriminator
from data_preprocessing import prepare_data_loaders

# --- 0. Focal Loss の定義 (新規追加) ---
class FocalLoss(nn.Module):
    """
    Focal Loss: 
    簡単なサンプルの損失を抑え、難しい（正解しにくい）サンプルの学習を重視する。
    alpha: クラスごとの重み (Tensor)
    gamma: 難易度調整パラメータ (大きいほど難しいサンプルを重視)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 標準的なCrossEntropy (reduction='none' で個別の損失を取得)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss) # 確率 p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 1. 設定とハイパーパラメータ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ★★★ チューニング設定 ★★★
BATCH_SIZE = 64
NUM_EPOCHS = 100

# 学習率 (TTUR)
LR_D = 0.000487
LR_G = 0.000232

# K-step (Dの更新回数)
K_STEPS = 2

# ラベルスムージング設定
LABEL_REAL = 0.95
LABEL_FAKE = 0.08

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

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NOISE_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 

# --- 2. データローダーの準備 ---
# ★ 変更点: balance_data=False にして、全てのデータを使用する ★
print("データをロード中... (ダウンサンプリングなし)")
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0,
    balance_data=False  # ★ 全データ使用
)

# --- ★ クラス重みの計算 (新規追加) ★ ---
print("クラスの重みを計算中...")
# TensorDatasetから全ラベルを取得
all_labels = train_loader.dataset.tensors[1].cpu().numpy()
label_counts = Counter(all_labels)
total_samples = len(all_labels)

# クラスID順に重みを計算 (サンプル数が少ないクラスほど重みを大きく)
# 重み = 全データ数 / (クラス数 * そのクラスのデータ数) などが一般的
# ここではシンプルに逆数を取り、最大値で正規化する等の調整を行います
class_weights = []
print("クラスごとのデータ数:")
for i in range(NUM_CLASSES):
    count = label_counts.get(i, 0)
    print(f"  Class {i}: {count}")
    # countが0の場合はエラー回避のため1として扱う（本来データがあるはずだが念のため）
    weight = total_samples / (count if count > 0 else 1)
    class_weights.append(weight)

# 重みをTensorに変換してデバイスへ転送
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
# 重みの正規化 (オプション: 平均が1になるようにする)
class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

print(f"計算されたクラス重み (Top 5): {class_weights_tensor[:5]}")

# --- 3. モデルの初期化 ---
if os.path.exists("embedding_matrix.pth"):
    print("事前学習済みEmbeddingをロードします...")
    embedding_weights = torch.load("embedding_matrix.pth").to(device)
else:
    print("注意: embedding_matrix.pth が見つかりません。ランダム初期化で開始します。")
    embedding_weights = None

# Generator (変更なし)
generator = Generator(VOCAB_SIZE, HIDDEN_DIM, NOISE_DIM, NUM_CLASSES, MAX_SEQUENCE_LENGTH).to(device)

# Discriminator (引数に重みを渡す)
discriminator = Discriminator(
    VOCAB_SIZE, 
    EMBEDDING_DIM, 
    HIDDEN_DIM, 
    NUM_CLASSES, 
    pretrained_embeddings=embedding_weights # ★ここに追加
).to(device)
# --- 4. 最適化アルゴリズムと損失関数 ---
optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

adversarial_loss = nn.BCEWithLogitsLoss() # 本物か偽物か

# ★ 変更点: Focal Loss を適用 (重み付き) ★
auxiliary_loss = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

# --- 5. 学習ループ ---
print(f"\n--- AC-GAN 学習開始 (Focal Loss適用) ---")
print(f"設定: K_STEPS={K_STEPS}, Real={LABEL_REAL}, Fake={LABEL_FAKE}")

for epoch in range(NUM_EPOCHS):
    for i, (real_seqs, real_labels) in enumerate(train_loader):
        
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(device)
        real_labels = real_labels.to(device)

        # ラベル定義
        valid = torch.full((batch_size, 1), LABEL_REAL, device=device)
        fake = torch.full((batch_size, 1), LABEL_FAKE, device=device)

        # ===============================================
        #  Train Discriminator
        # ===============================================
        optimizer_d.zero_grad()

        # 1. 本物のデータを判定
        pred_validity, pred_class = discriminator(real_seqs)
        d_loss_real_val = adversarial_loss(pred_validity, valid)
        d_loss_real_class = auxiliary_loss(pred_class, real_labels) # Focal Loss
        d_loss_real = d_loss_real_val + d_loss_real_class

        # 2. 偽物のデータを判定
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
        #  Train Generator (K_STEPS回に1回)
        # ===============================================
        g_loss_val = 0.0

        if i % K_STEPS == 0:
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

        if i % 100 == 0:
            print(
                f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss_val:.4f}]"
            )

    print(f"--- Epoch {epoch+1} 完了 ---")
    
# --- 保存 ---
print("\n--- 学習完了。モデルを保存します ---")
torch.save(generator.state_dict(), "acgan_generator_focal.pth")
torch.save(discriminator.state_dict(), "acgan_discriminator_focal.pth")
print("保存完了")
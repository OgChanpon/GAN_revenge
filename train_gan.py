# train_gan.py (Filtered Data + Focal + Spectral + Word2Vec)

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
import torch.nn.functional as F
from collections import Counter

from models import Generator, Discriminator
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

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

BATCH_SIZE = 64
NUM_EPOCHS = 100 
LR_D = 0.000487
LR_G = 0.000232
K_STEPS = 2
LABEL_REAL = 0.95
LABEL_FAKE = 0.08

# ★ 重要: フィルタ済みの辞書をロード
WORD_TO_INT_PATH = 'word_to_int.pkl' # これは共通でOK
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl' # ★ここを変更

if not os.path.exists(CATEGORY_TO_ID_PATH):
    print("エラー: フィルタ済みの辞書が見つかりません。python3 filter_dataset.py を実行しましたか？")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open(CATEGORY_TO_ID_PATH, "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)
print(f"クラス数: {NUM_CLASSES} (フィルタ済み)")

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NOISE_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 

# --- データロード ---
print("データをロード中...")
# data_preprocessing.py 側で 'train_dataset_filtered.pkl' を読むように書き換えている前提
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0,
    balance_data=False  # ★ ダウンサンプリング廃止
)

# --- クラス重み計算 ---
print("クラス重みを計算中...")
all_labels = train_loader.dataset.tensors[1].cpu().numpy()
label_counts = Counter(all_labels)
total_samples = len(all_labels)
class_weights = []
for i in range(NUM_CLASSES):
    count = label_counts.get(i, 0)
    # フィルタリングしたので count=0 のクラスは無いはずだが安全策
    weight = total_samples / (count if count > 0 else 1)
    class_weights.append(weight)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

# --- Word2Vec (Step 4) ---
if os.path.exists("embedding_matrix.pth"):
    print("事前学習済みEmbedding (Word2Vec) をロードします...")
    embedding_weights = torch.load("embedding_matrix.pth").to(device)
else:
    print("注意: embedding_matrix.pth がありません。ランダム初期化します。")
    embedding_weights = None

# --- モデル初期化 ---
# models.py は前回お渡しした「Spectral Norm強化版 (Step 3)」を使用
generator = Generator(VOCAB_SIZE, HIDDEN_DIM, NOISE_DIM, NUM_CLASSES, MAX_SEQUENCE_LENGTH).to(device)
discriminator = Discriminator(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, 
    pretrained_embeddings=embedding_weights
).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

adversarial_loss = nn.BCEWithLogitsLoss()
auxiliary_loss = FocalLoss(alpha=class_weights_tensor, gamma=2.0) # Step 2

# --- 学習ループ ---
print(f"\n--- AC-GAN Training Start (Filtered Data) ---")
for epoch in range(NUM_EPOCHS):
    for i, (real_seqs, real_labels) in enumerate(train_loader):
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(device)
        real_labels = real_labels.to(device)

        valid = torch.full((batch_size, 1), LABEL_REAL, device=device)
        fake = torch.full((batch_size, 1), LABEL_FAKE, device=device)

        # Train Discriminator
        optimizer_d.zero_grad()
        
        pred_validity, pred_class = discriminator(real_seqs)
        d_loss_real = adversarial_loss(pred_validity, valid) + auxiliary_loss(pred_class, real_labels)

        z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        fake_logits = generator(z, gen_labels)
        fake_seqs_discrete = torch.argmax(fake_logits, dim=2)
        
        pred_validity_fake, _ = discriminator(fake_seqs_discrete.detach())
        d_loss_fake = adversarial_loss(pred_validity_fake, fake)

        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        g_loss_val = 0.0
        if i % K_STEPS == 0:
            optimizer_g.zero_grad()
            
            fake_probs = F.softmax(fake_logits, dim=2)
            embed_matrix = discriminator.embedding.weight
            soft_input = torch.matmul(fake_probs, embed_matrix)
            
            pred_validity, pred_class = discriminator(None, soft_input=soft_input)
            
            # Generatorは「騙せたか(valid=1)」かつ「指定クラスに分類されたか」を学習
            g_loss = adversarial_loss(pred_validity, torch.full((batch_size, 1), 1.0, device=device)) + \
                     auxiliary_loss(pred_class, gen_labels)
            
            g_loss.backward()
            optimizer_g.step()
            g_loss_val = g_loss.item()

        if i % 100 == 0:
            print(f"[Epoch {epoch+1}] D loss: {d_loss.item():.4f} G loss: {g_loss_val:.4f}")

    print(f"Epoch {epoch+1} done.")

# 保存
print("保存中...")
torch.save(generator.state_dict(), "acgan_generator_filtered.pth")
torch.save(discriminator.state_dict(), "acgan_discriminator_filtered.pth")
print("完了")
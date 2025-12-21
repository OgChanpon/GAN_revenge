# train_transformer_acgan.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import numpy as np
from collections import Counter

from models import Generator
from models_transformer_acgan import TransformerDiscriminator # 新しいD
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

# ★ フィルタ済みデータを使用 ★
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl'

# Optunaベースのパラメータ
BATCH_SIZE = 32
LR_D = 5e-5          # Transformerなので低めに
LR_G = 0.0002
GAMMA = 4.729
NUM_EPOCHS = 50
K_STEPS = 2

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NOISE_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 
NUM_HEADS = 8
NUM_LAYERS = 2

# --- データ準備 ---
if not os.path.exists(CATEGORY_TO_ID_PATH):
    print("エラー: フィルタ済み辞書が見つかりません。")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f: vocab_size = len(pickle.load(f))
with open(CATEGORY_TO_ID_PATH, "rb") as f: num_classes = len(pickle.load(f))

print("データをロード中...")
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, max_length=MAX_SEQUENCE_LENGTH, validation_split=0.0, balance_data=False
)

# クラス重み
all_labels = train_loader.dataset.tensors[1].cpu().numpy()
counts = Counter(all_labels)
weights = [len(all_labels) / (counts[i] if counts[i]>0 else 1) for i in range(num_classes)]
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
class_weights_tensor /= class_weights_tensor.mean()

# Embeddingロード
embedding_weights = None
if os.path.exists("embedding_matrix.pth"):
    embedding_weights = torch.load("embedding_matrix.pth").to(device)

# --- モデル初期化 ---
print("Transformer-AC-GAN モデル初期化...")
generator = Generator(vocab_size, HIDDEN_DIM, NOISE_DIM, num_classes, MAX_SEQUENCE_LENGTH).to(device)

# ★ ここが新しいDiscriminator
discriminator = TransformerDiscriminator(
    vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes, 
    num_heads=NUM_HEADS, num_layers=NUM_LAYERS, max_len=MAX_SEQUENCE_LENGTH,
    pretrained_embeddings=embedding_weights
).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999))

adversarial_loss = nn.BCEWithLogitsLoss()
auxiliary_loss = FocalLoss(alpha=class_weights_tensor, gamma=GAMMA)

# --- 学習ループ ---
print("\n--- Transformer-AC-GAN Training Start ---")

for epoch in range(NUM_EPOCHS):
    for i, (real_seqs, real_labels) in enumerate(train_loader):
        bs = real_seqs.size(0)
        real_seqs, real_labels = real_seqs.to(device), real_labels.to(device)
        valid = torch.full((bs, 1), 0.95, device=device)
        fake = torch.full((bs, 1), 0.08, device=device)

        # --- Train Discriminator ---
        optimizer_d.zero_grad()
        
        # Real
        p_val, p_cls = discriminator(real_seqs)
        loss_real_adv = adversarial_loss(p_val, valid)
        loss_real_cls = auxiliary_loss(p_cls, real_labels)
        
        # Fake
        z = torch.randn(bs, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, num_classes, (bs,), device=device)
        fake_logits = generator(z, gen_labels)
        fake_seqs = torch.argmax(fake_logits, dim=2)
        
        p_val_fake, _ = discriminator(fake_seqs.detach())
        loss_fake_adv = adversarial_loss(p_val_fake, fake)
        
        loss_d = (loss_real_adv + loss_real_cls + loss_fake_adv) / 2
        loss_d.backward()
        optimizer_d.step()

        # --- Train Generator ---
        if i % K_STEPS == 0:
            optimizer_g.zero_grad()
            
            fake_probs = F.softmax(fake_logits, dim=2)
            # Embedding層の重みを使ってSoft Inputを作る
            if hasattr(discriminator.embedding, 'weight'):
                embed_mat = discriminator.embedding.weight
            else:
                # nn.Embedding以外の場合の予備（通常ここには来ない）
                embed_mat = discriminator.module.embedding.weight 
            
            soft_input = torch.matmul(fake_probs, embed_mat)
            
            p_val_g, p_cls_g = discriminator(None, soft_input=soft_input)
            
            loss_g = adversarial_loss(p_val_g, torch.full((bs, 1), 1.0, device=device)) + \
                     auxiliary_loss(p_cls_g, gen_labels)
            
            loss_g.backward()
            optimizer_g.step()

    if (epoch+1) % 5 == 0:
        print(f"[Epoch {epoch+1}/{NUM_EPOCHS}] D Loss: {loss_d.item():.4f}")

# 保存
print("保存中...")
torch.save(generator.state_dict(), "transformer_acgan_generator.pth")
torch.save(discriminator.state_dict(), "transformer_acgan_discriminator.pth")
print("完了")
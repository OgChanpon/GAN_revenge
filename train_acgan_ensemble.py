# train_acgan_ensemble.py (AC-GAN Full Ensemble)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import random
import numpy as np
from collections import Counter

# --- 必要なモジュール ---
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
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# --- 設定 (Optuna Best Params) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")

# ★ Optunaで見つけたパラメータ ★
BATCH_SIZE = 32
LR_D = 5.177e-05     # Discriminatorの学習率
LR_G = 0.000232      # Generatorは標準的ままでOK（あるいはDに合わせる）
GAMMA = 4.729        # Focal Loss
WEIGHT_DECAY = 5.908e-05 

NUM_EPOCHS = 50      # 各モデルの学習Epoch数
NUM_MODELS = 5       # 作るモデルの数（アンサンブル数）
K_STEPS = 2          # Dの更新回数

# パス
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl' # フィルタ済み
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NOISE_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 

# --- シード固定関数 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# --- データ準備 ---
if not os.path.exists(CATEGORY_TO_ID_PATH):
    print("エラー: フィルタ済み辞書がありません。")
    exit()

with open(WORD_TO_INT_PATH, "rb") as f: vocab_size = len(pickle.load(f))
with open(CATEGORY_TO_ID_PATH, "rb") as f: num_classes = len(pickle.load(f))

print("データをロード中...")
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH, 
    validation_split=0.0, # フルデータ使用
    balance_data=False
)

# クラス重み
all_labels = train_loader.dataset.tensors[1].cpu().numpy()
counts = Counter(all_labels)
weights = [len(all_labels) / (counts[i] if counts[i]>0 else 1) for i in range(num_classes)]
class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
class_weights_tensor /= class_weights_tensor.mean()

# Embedding読み込み
embedding_weights = None
if os.path.exists("embedding_matrix.pth"):
    embedding_weights = torch.load("embedding_matrix.pth").to(device)

# --- アンサンブル AC-GAN 学習ループ ---
print(f"\n=== AC-GAN Ensemble Training Start ({NUM_MODELS} models) ===")

for model_idx in range(NUM_MODELS):
    # シードを変えて初期値をばらつかせる
    current_seed = 42 + model_idx
    set_seed(current_seed)
    
    print(f"\n[Model {model_idx+1}/{NUM_MODELS}] Seed: {current_seed} Training...")
    
    # モデル初期化
    generator = Generator(vocab_size, HIDDEN_DIM, NOISE_DIM, num_classes, MAX_SEQUENCE_LENGTH).to(device)
    discriminator = Discriminator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes, pretrained_embeddings=embedding_weights).to(device)
    
    # Optimizer
    optimizer_g = optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.999), weight_decay=WEIGHT_DECAY)
    
    adversarial_loss = nn.BCEWithLogitsLoss()
    auxiliary_loss = FocalLoss(alpha=class_weights_tensor, gamma=GAMMA)

    # 学習ループ
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
            loss_real = adversarial_loss(p_val, valid) + auxiliary_loss(p_cls, real_labels)
            
            # Fake
            z = torch.randn(bs, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
            gen_labels = torch.randint(0, num_classes, (bs,), device=device)
            fake_logits = generator(z, gen_labels)
            fake_seqs = torch.argmax(fake_logits, dim=2)
            
            p_val_fake, _ = discriminator(fake_seqs.detach())
            loss_fake = adversarial_loss(p_val_fake, fake)
            
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator ---
            if i % K_STEPS == 0:
                optimizer_g.zero_grad()
                
                # Gumbel-Softmax的な勾配伝搬のための工夫 (Soft Input)
                fake_probs = F.softmax(fake_logits, dim=2)
                embed_mat = discriminator.embedding.weight
                soft_input = torch.matmul(fake_probs, embed_mat)
                
                p_val_g, p_cls_g = discriminator(None, soft_input=soft_input)
                
                loss_g = adversarial_loss(p_val_g, torch.full((bs, 1), 1.0, device=device)) + \
                         auxiliary_loss(p_cls_g, gen_labels)
                
                loss_g.backward()
                optimizer_g.step()

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS} done.")

    # 保存
    save_name = f"acgan_ensemble_discriminator_{model_idx}.pth"
    torch.save(discriminator.state_dict(), save_name)
    print(f"  Saved: {save_name}")

print("\n全AC-GANモデルの学習完了！")
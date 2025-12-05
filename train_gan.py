# train_gan.py (AC-GAN 学習スクリプト)

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

# ハイパーパラメータ
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NOISE_DIM = 100
MAX_SEQUENCE_LENGTH = 1000 # tokenka.pyと合わせる
BATCH_SIZE = 64
NUM_EPOCHS = 50 # データが多いので50エポックでも十分学習するはず
LR = 0.0002

# --- 2. データローダーの準備 ---
# data_preprocessing.py の関数を利用
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0 # GANの学習では全データを訓練に使うことが多い
)

# --- 3. モデルの初期化 ---
generator = Generator(VOCAB_SIZE, HIDDEN_DIM, NOISE_DIM, NUM_CLASSES, MAX_SEQUENCE_LENGTH).to(device)
discriminator = Discriminator(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)

# --- 4. 最適化アルゴリズムと損失関数 ---
optimizer_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

adversarial_loss = nn.BCEWithLogitsLoss() # 本物か偽物か
auxiliary_loss = nn.CrossEntropyLoss()    # どのクラスか

# --- 5. 学習ループ ---
print("\n--- AC-GAN 学習開始 ---")

for epoch in range(NUM_EPOCHS):
    for i, (real_seqs, real_labels) in enumerate(train_loader):
        
        batch_size = real_seqs.size(0)
        real_seqs = real_seqs.to(device)
        real_labels = real_labels.to(device)

        # 正解ラベル (本物=1, 偽物=0)
        # Label Smoothing (本物を0.9にする) を適用してDの過信を防ぐ
        valid = torch.full((batch_size, 1), 0.9, device=device)
        fake = torch.full((batch_size, 1), 0.0, device=device)

        # -----------------
        #  Train Discriminator
        # -----------------
        optimizer_d.zero_grad()

        # 1. 本物のデータを判定
        pred_validity, pred_class = discriminator(real_seqs)
        d_loss_real_val = adversarial_loss(pred_validity, valid)
        d_loss_real_class = auxiliary_loss(pred_class, real_labels)
        d_loss_real = d_loss_real_val + d_loss_real_class

        # 2. 偽物のデータを判定
        # ノイズとランダムなラベルを生成
        z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        
        # Generatorで偽造 (Dの学習時は、勾配を切るためにdetachするか、離散化する)
        # ここでは離散化(argmax)して「ハードな偽物」としてDに見せる
        fake_logits = generator(z, gen_labels)
        fake_seqs_discrete = torch.argmax(fake_logits, dim=2)
        
        pred_validity_fake, pred_class_fake = discriminator(fake_seqs_discrete)
        
        # 偽物に対する損失 (偽物と見抜ければOK。クラス分類は求めても求めなくても良いが、今回はValidityのみに注目)
        d_loss_fake_val = adversarial_loss(pred_validity_fake, fake)
        d_loss_fake = d_loss_fake_val 

        # Dの合計損失
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_d.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_g.zero_grad()

        # もう一度ノイズから生成 (Gに勾配を流すため)
        z = torch.randn(batch_size, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
        
        fake_logits = generator(z, gen_labels)
        
        # ★ソフトな埋め込み (Soft Embedding) ★
        # argmaxの代わりに、確率分布を使ってDの埋め込み層を通過させる
        fake_probs = F.softmax(fake_logits, dim=2)
        embed_matrix = discriminator.embedding.weight
        # (Batch, SeqLen, Vocab) x (Vocab, EmbDim) -> (Batch, SeqLen, EmbDim)
        soft_input = torch.matmul(fake_probs, embed_matrix)
        
        # Dに入力 (soft_inputモード)
        pred_validity, pred_class = discriminator(None, soft_input=soft_input)

        # Gの損失
        # 1. 騙したい (Validity = 1)
        g_loss_validity = adversarial_loss(pred_validity, torch.full((batch_size, 1), 1.0, device=device))
        # 2. 指定したクラスだと認識させたい
        g_loss_class = auxiliary_loss(pred_class, gen_labels)
        
        g_loss = g_loss_validity + g_loss_class
        g_loss.backward()
        optimizer_g.step()

        if i % 50 == 0:
            print(
                f"[Epoch {epoch+1}/{NUM_EPOCHS}] [Batch {i}/{len(train_loader)}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

    # --- エポックごとの生成サンプル確認 ---
    print(f"--- Epoch {epoch+1} 生成サンプル ---")
    with torch.no_grad():
        # クラス0 (最初のクラス) のサンプルを生成してみる
        test_z = torch.randn(1, MAX_SEQUENCE_LENGTH, NOISE_DIM).to(device)
        test_label = torch.tensor([0], device=device) # クラス0を指定
        
        gen_logits = generator(test_z, test_label)
        gen_seq = torch.argmax(gen_logits, dim=2).squeeze().cpu().numpy()
        
        # IDを単語に戻す
        int_to_word = {v: k for k, v in word_to_int.items()}
        apis = [int_to_word.get(idx, "_UNK_") for idx in gen_seq if idx != 0] # PAD除く
        print(f"Class 0 Sample: {apis[:10]} ...") # 先頭10個だけ表示

# --- 保存 ---
print("\n--- 学習完了。モデルを保存します ---")
torch.save(generator.state_dict(), "acgan_generator.pth")
torch.save(discriminator.state_dict(), "acgan_discriminator.pth")
print("保存完了: acgan_generator.pth, acgan_discriminator.pth")
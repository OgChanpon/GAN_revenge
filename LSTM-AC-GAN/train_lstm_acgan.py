# train_lstm_acgan.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
from models_lstm_acgan import LSTMGenerator, LSTMDiscriminator
from data_preprocessing import prepare_data_loaders

# --- Optuna指定ハイパーパラメータ ---
K_STEP = 1
LR_D = 0.0004
LR_G = 0.00014
LABEL_REAL = 0.95
LABEL_FAKE = 0.12
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
# --------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 100 # AC-GANなので多めに
NOISE_DIM = 100
MAX_LEN = 1000

WORD_TO_INT = 'word_to_int.pkl'
CAT_TO_ID = 'category_to_id_filtered_full.pkl'

def main():
    print(f"=== 実験3: LSTM-AC-GAN (指定パラメータ) 学習開始 ===")
    
    with open(WORD_TO_INT, 'rb') as f: vocab_size = len(pickle.load(f))
    with open(CAT_TO_ID, 'rb') as f: num_classes = len(pickle.load(f))

    # データロード
    train_loader, _ = prepare_data_loaders(
        batch_size=BATCH_SIZE, max_length=MAX_LEN, validation_split=0.0, balance_data=False
    )

    # モデル初期化
    generator = LSTMGenerator(vocab_size, HIDDEN_DIM, NOISE_DIM, num_classes, MAX_LEN).to(DEVICE)
    discriminator = LSTMDiscriminator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(DEVICE)

    # Optimizer
    opt_g = optim.Adam(generator.parameters(), lr=LR_G)
    opt_d = optim.Adam(discriminator.parameters(), lr=LR_D)
    
    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_cls = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        for i, (real_seqs, real_labels) in enumerate(train_loader):
            bs = real_seqs.size(0)
            real_seqs, real_labels = real_seqs.to(DEVICE), real_labels.to(DEVICE)
            
            # ラベル定義
            valid = torch.full((bs, 1), LABEL_REAL, device=DEVICE)
            fake = torch.full((bs, 1), LABEL_FAKE, device=DEVICE)
            
            # --- Train Discriminator ---
            opt_d.zero_grad()
            
            # Real
            p_val, p_cls = discriminator(real_seqs)
            loss_real = criterion_adv(p_val, valid) + criterion_cls(p_cls, real_labels)
            
            # Fake
            z = torch.randn(bs, MAX_LEN, NOISE_DIM).to(DEVICE)
            gen_labels = torch.randint(0, num_classes, (bs,), device=DEVICE)
            fake_logits = generator(z, gen_labels)
            fake_seqs = torch.argmax(fake_logits, dim=2)
            
            p_val_fake, p_cls_fake = discriminator(fake_seqs.detach())
            loss_fake = criterion_adv(p_val_fake, fake) + criterion_cls(p_cls_fake, gen_labels)
            
            loss_d = (loss_real + loss_fake) / 2
            loss_d.backward()
            opt_d.step()
            
            # --- Train Generator (K_STEP=1) ---
            if i % K_STEP == 0:
                opt_g.zero_grad()
                
                # Gumbel Softmax的な処理 (Embeddingを通すため)
                fake_probs = F.softmax(fake_logits, dim=2)
                embed_weight = discriminator.embedding.weight
                soft_input = torch.matmul(fake_probs, embed_weight)
                
                p_val_g, p_cls_g = discriminator(None, soft_input=soft_input)
                
                # Generatorは「本物(1.0)と判定させたい」「指定クラスに分類させたい」
                # ここではvalid(0.95)ではなく完全な本物ラベル(1.0)を目指すのが一般的だが
                # 指定パラメータの意図により LABEL_REAL(0.95) を使うこともあり得る。
                # ここでは標準的に LABEL_REAL を目指すとする。
                loss_g = criterion_adv(p_val_g, valid) + criterion_cls(p_cls_g, gen_labels)
                
                loss_g.backward()
                opt_g.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Done.")

    # Discriminator保存
    torch.save(discriminator.state_dict(), "lstm_acgan_discriminator_final.pth")
    print("学習完了。LSTM-AC-GAN Discriminator保存済み。")

if __name__ == '__main__':
    main()
# train_gan_optuna.py (BERT-AC-GAN 最適化版)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import os
import optuna
from optuna.trial import TrialState

# 必要なモジュール
from models import Generator, Discriminator
from data_preprocessing import prepare_data_loaders

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # Transformerはメモリ消費が大きいので安全策で32
MAX_SEQUENCE_LENGTH = 1000
N_TRIALS = 50       
EPOCHS_PER_TRIAL = 15 # Transformerは学習が遅いので少し減らす

# 辞書ロード
with open("word_to_int.pkl", "rb") as f:
    word_to_int = pickle.load(f)
VOCAB_SIZE = len(word_to_int)

with open("category_to_id.pkl", "rb") as f:
    category_to_id = pickle.load(f)
NUM_CLASSES = len(category_to_id)

# テストデータ
with open("test_dataset.pkl", "rb") as f:
    test_data = pickle.load(f)

# データローダー
train_loader, _ = prepare_data_loaders(
    batch_size=BATCH_SIZE, 
    max_length=MAX_SEQUENCE_LENGTH,
    validation_split=0.0
)

# --- 評価関数 ---
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq, label in test_data:
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            _, class_logits = model(seq_tensor)
            pred = torch.argmax(class_logits, dim=1).item()
            if pred == label:
                correct += 1
            total += 1
    return correct / total

# --- 目的関数 ---
def objective(trial):
    # ==========================
    # 1. パラメータの探索空間 (Transformer仕様)
    # ==========================
    
    # モデル構造
    # d_model (hidden_dim) は nhead で割り切れる必要がある
    # [128, 256, 512] はすべて 2, 4, 8 で割り切れるので安全
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512]) 
    embedding_dim = hidden_dim # TransformerではEmbed=Hiddenにするのが定石
    
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])
    num_layers = trial.suggest_int("num_layers", 2, 4) # 深すぎるとGANは学習しにくい
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    noise_dim = 100
    
    # 学習率 (Transformerは少し低めが良い傾向)
    lr_d = trial.suggest_float("lr_d", 5e-5, 5e-4, log=True)
    g_lr_ratio = trial.suggest_float("g_lr_ratio", 0.1, 1.2)
    lr_g = lr_d * g_lr_ratio
    
    # 更新回数
    k_steps = trial.suggest_int("k_steps", 1, 3)
    g_steps = trial.suggest_int("g_steps", 1, 3)
    
    # ラベルスムージング
    label_real = trial.suggest_float("label_real", 0.85, 1.0)
    label_fake = trial.suggest_float("label_fake", 0.0, 0.2)
    
    # ==========================
    # 2. モデル構築
    # ==========================
    generator = Generator(VOCAB_SIZE, hidden_dim, noise_dim, NUM_CLASSES, MAX_SEQUENCE_LENGTH).to(device)
    
    # DiscriminatorにOptunaのパラメータを渡す
    discriminator = Discriminator(
        vocab_size=VOCAB_SIZE, 
        embedding_dim=embedding_dim, 
        hidden_dim=hidden_dim, 
        num_classes=NUM_CLASSES,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    
    adversarial_loss = nn.BCEWithLogitsLoss()
    auxiliary_loss = nn.CrossEntropyLoss()
    
    # ==========================
    # 3. 学習ループ
    # ==========================
    for epoch in range(EPOCHS_PER_TRIAL):
        discriminator.train()
        generator.train()
        
        for i, (real_seqs, real_labels) in enumerate(train_loader):
            batch_len = real_seqs.size(0)
            real_seqs = real_seqs.to(device)
            real_labels = real_labels.to(device)
            
            valid = torch.full((batch_len, 1), label_real, device=device)
            fake = torch.full((batch_len, 1), label_fake, device=device)
            
            # --- Train D ---
            optimizer_d.zero_grad()
            
            pred_validity, pred_class = discriminator(real_seqs)
            d_loss_real = adversarial_loss(pred_validity, valid) + auxiliary_loss(pred_class, real_labels)
            
            z = torch.randn(batch_len, MAX_SEQUENCE_LENGTH, noise_dim).to(device)
            gen_labels = torch.randint(0, NUM_CLASSES, (batch_len,), device=device)
            fake_seqs = torch.argmax(generator(z, gen_labels), dim=2)
            
            pred_validity_fake, _ = discriminator(fake_seqs.detach())
            d_loss_fake = adversarial_loss(pred_validity_fake, fake)
            
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # --- Train G ---
            if i % k_steps == 0:
                for _ in range(g_steps):
                    optimizer_g.zero_grad()
                    z = torch.randn(batch_len, MAX_SEQUENCE_LENGTH, noise_dim).to(device)
                    gen_labels = torch.randint(0, NUM_CLASSES, (batch_len,), device=device)
                    
                    fake_logits = generator(z, gen_labels)
                    fake_probs = F.softmax(fake_logits, dim=2)
                    soft_input = torch.matmul(fake_probs, discriminator.embedding.weight)
                    
                    pred_validity, pred_class = discriminator(None, soft_input=soft_input)
                    
                    valid_target = torch.full((batch_len, 1), 1.0, device=device)
                    g_loss = adversarial_loss(pred_validity, valid_target) + auxiliary_loss(pred_class, gen_labels)
                    g_loss.backward()
                    optimizer_g.step()

        # --- 評価と枝刈り ---
        accuracy = evaluate_model(discriminator)
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

if __name__ == "__main__":
    print(f"OptunaによるBERT-AC-GAN最適化を開始します (試行回数: {N_TRIALS})")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n==================================")
    print(f"Best Trial Accuracy: {study.best_value:.4f}")
    print("Best Params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    study.trials_dataframe().to_csv("optuna_results_bert.csv")
    print("結果を optuna_results_bert.csv に保存しました。")
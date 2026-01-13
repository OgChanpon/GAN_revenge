# data_preprocessing.py

import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

def prepare_data_loaders(batch_size=64, max_length=1000, validation_split=0.1, balance_data=False):
    """
    balance_data=True にすると、訓練データを最小クラス数に合わせてダウンサンプリングします。
    """
    
    # 1. データの読み込み
    with open("train_dataset_filtered_full.pkl", "rb") as f:
        train_data = pickle.load(f)
        
    # (シーケンス, ラベル) のリストを分解
    sequences = [item[0] for item in train_data]
    labels = [item[1] for item in train_data]
    
    # --- ★ ダウンサンプリング処理 (ここが追加箇所) ★ ---
    if balance_data:
        print("\n[Info] クラス不均衡の調整（ダウンサンプリング）を開始します...")
        
        # クラスごとのデータ数をカウント
        label_counts = Counter(labels)
        min_count = min(label_counts.values())
        print(f"最小クラスのデータ数: {min_count}")
        print("各クラスのデータ数をこの数に揃えます。")
        
        # クラスごとにインデックスを分ける
        indices_by_class = {label: [] for label in label_counts}
        for idx, label in enumerate(labels):
            indices_by_class[label].append(idx)
            
        # 各クラスから min_count 個だけランダムに選ぶ
        balanced_indices = []
        np.random.seed(42) # 再現性のため固定
        for label in indices_by_class:
            chosen_indices = np.random.choice(indices_by_class[label], min_count, replace=False)
            balanced_indices.extend(chosen_indices)
            
        # 選んだインデックスだけでデータを再構築
        # (リスト内包表記で抽出)
        sequences = [sequences[i] for i in balanced_indices]
        labels = [labels[i] for i in balanced_indices]
        
        print(f"調整後の総データ数: {len(sequences)}")
        print(f"クラスごとのデータ数: {Counter(labels)}\n")
    # ----------------------------------------------------

    # 2. パディング (長さを揃える)
    sequences_padded = []
    for seq in sequences:
        if len(seq) > max_length:
            sequences_padded.append(seq[:max_length])
        else:
            sequences_padded.append(seq + [0] * (max_length - len(seq)))
            
    X = torch.tensor(sequences_padded, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    # 3. 訓練データと検証データに分割
    # validation_split が 0.0 の場合は分割しない
    if validation_split > 0.0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    else:
        # 検証データなし（全データを訓練に使う）
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader, None
# data_preprocessing.py (AC-GAN用)

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

# --- 1. PyTorch用のカスタムデータセットクラス ---
class MalwareSequenceDataset(Dataset):
    def __init__(self, data_list):
        # data_list: [(sequence, label), ...] のリスト
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, label = self.data[idx]
        
        # シーケンスとラベルをPyTorchテンソルに変換
        # sequence: 長さ可変のリスト -> LongTensor
        # label: 整数 -> LongTensor (CrossEntropyLoss用)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# --- 2. バッチ作成時のパディング関数 ---
def collate_fn(batch, max_length):
    sequences = []
    labels = []
    
    for seq, lbl in batch:
        labels.append(lbl)
        
        # パディング処理 (0埋め)
        if len(seq) > max_length:
            sequences.append(seq[:max_length])
        else:
            pad_size = max_length - len(seq)
            # torch.cat で結合
            sequences.append(torch.cat([seq, torch.zeros(pad_size, dtype=torch.long)]))

    # バッチとしてまとめる
    # sequences: (Batch, MaxLen)
    # labels: (Batch)
    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)

# --- 3. データローダー準備関数 ---
def prepare_data_loaders(batch_size=64, max_length=1000, validation_split=0.0):
    """
    tokenka.py で作成された train_dataset.pkl を読み込み、DataLoaderを返す。
    GANの学習では通常、全データを学習に使うため validation_split はデフォルト 0.0。
    """
    print("Step: データセットの読み込み...")
    
    if not os.path.exists("train_dataset.pkl"):
        print("エラー: train_dataset.pkl が見つかりません。先に tokenka.py を実行してください。")
        return None, None

    with open("train_dataset.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    print(f"読み込み完了。データ数: {len(train_data)} 件")

    # データセット作成
    dataset = MalwareSequenceDataset(train_data)
    
    # DataLoader作成
    # collate_fn に max_length を渡すためのラッパー
    collate_wrapper = lambda batch: collate_fn(batch, max_length)
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, # GAN学習ではシャッフル必須
        collate_fn=collate_wrapper,
        drop_last=True # バッチサイズが半端だとエラーの元になるので切り捨てる
    )
    
    print(f"DataLoader作成完了。バッチサイズ: {batch_size}, シーケンス長: {max_length}")
    
    # 戻り値は (train_loader, val_loader) だが、今回は val_loader は None
    return train_loader, None

# --- 動作確認用 ---
if __name__ == "__main__":
    loader, _ = prepare_data_loaders(batch_size=4, max_length=100)
    if loader:
        seqs, labels = next(iter(loader))
        print("\n--- バッチ確認 ---")
        print(f"Sequences shape: {seqs.shape}") # (4, 100)
        print(f"Labels shape: {labels.shape}")   # (4)
        print(f"Labels: {labels}")
# evaluate_acgan.py (AC-GAN 分類精度評価)

import torch
import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# models.py から Discriminator をインポート
from models import Discriminator

# --- 設定 ---
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl'
#MODEL_PATH = 'acgan_discriminator.pth'
#MODEL_PATH = 'acgan_discriminator_focal.pth'
#MODEL_PATH = 'acgan_discriminator_finetuned.pth'
MODEL_PATH = 'acgan_discriminator_best.pth'
TEST_DATA_PATH = 'test_dataset_filtered.pkl'

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
# AC-GANのDiscriminatorはnum_classesが必要
with open(CATEGORY_TO_ID_PATH, 'rb') as f:
    cat_to_id = pickle.load(f)
NUM_CLASSES = len(cat_to_id)
print(f"クラス数: {NUM_CLASSES}")

def load_model(device, vocab_size):
    model = Discriminator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデル {MODEL_PATH} が見つかりません。")
        exit()
    model.to(device)
    model.eval()
    return model

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # 辞書ロード
    with open(WORD_TO_INT_PATH, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # データロード
    with open(TEST_DATA_PATH, 'rb') as f:
        test_data = pickle.load(f) # [(seq, label), ...]
    
    print(f"テストデータ数: {len(test_data)}")

    # モデルロード
    model = load_model(device, vocab_size)

    # 評価ループ
    y_true = []
    y_pred = []

    print("評価中...")
    with torch.no_grad():
        for seq, label in test_data:
            # (SeqLen,) -> (1, SeqLen)
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            
            # AC-GANのDは (validity, class_logits) を返す
            _, class_logits = model(seq_tensor)
            
            pred_label = torch.argmax(class_logits, dim=1).item()
            
            y_true.append(label)
            y_pred.append(pred_label)

    # 結果表示
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== 最終評価結果 (AC-GAN Discriminator) ===")
    print(f"Accuracy (正解率): {acc:.4f} ({acc*100:.2f}%)")
    
    # クラス名取得
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    target_names = [id_to_cat[i] for i in range(NUM_CLASSES)]
    
    print("\n--- 詳細レポート ---")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # 混同行列の表示 (テキスト)
    print("\n--- 混同行列 ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == '__main__':
    evaluate()
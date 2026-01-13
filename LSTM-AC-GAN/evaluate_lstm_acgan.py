# evaluate_lstm_acgan.py

import torch
import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models_lstm_acgan import LSTMDiscriminator

# --- 設定 ---
MODEL_PATH = "lstm_acgan_discriminator_final.pth"  # 学習で保存したファイル名
TEST_DATA_PATH = 'test_dataset_filtered_full.pkl'
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered_full.pkl'

# 指定されたパラメータ
EMBEDDING_DIM = 256
HIDDEN_DIM = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"Loading data...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: モデルファイル {MODEL_PATH} が見つかりません。")
        return

    with open(WORD_TO_INT_PATH, 'rb') as f: vocab_size = len(pickle.load(f))
    with open(CATEGORY_TO_ID_PATH, 'rb') as f: 
        cat_to_id = pickle.load(f)
        num_classes = len(cat_to_id)
    with open(TEST_DATA_PATH, 'rb') as f: test_data = pickle.load(f)

    # モデルロード
    model = LSTMDiscriminator(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    y_true = []
    y_pred = []

    print(f"Evaluating LSTM-AC-GAN...")
    with torch.no_grad():
        for seq, label in test_data:
            # Padding
            if len(seq) < 1000:
                seq = seq + [0] * (1000 - len(seq))
            else:
                seq = seq[:1000]

            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            
            # 推論 (Discriminatorの戻り値は val, logits)
            _, logits = model(seq_tensor)
            pred = torch.argmax(logits, dim=1).item()
            
            y_true.append(label)
            y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== LSTM-AC-GAN Result ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    names = [id_to_cat[i] for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate()
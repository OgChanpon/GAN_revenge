# evaluate_ensemble.py

import torch
import pickle
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models import Discriminator

# --- 設定 ---
MODEL_FILES = [f"finetuned_ensemble_{i}.pth" for i in range(5)] # 5つのモデル
TEST_DATA_PATH = 'test_dataset_filtered.pkl'
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"Loading data...")
    with open(WORD_TO_INT_PATH, 'rb') as f: vocab_size = len(pickle.load(f))
    with open(CATEGORY_TO_ID_PATH, 'rb') as f: 
        cat_to_id = pickle.load(f)
        num_classes = len(cat_to_id)
    with open(TEST_DATA_PATH, 'rb') as f: test_data = pickle.load(f)

    # 5つのモデルをロード
    models = []
    print("Loading ensemble models...")
    for path in MODEL_FILES:
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping.")
            continue
        m = Discriminator(vocab_size, 256, 512, num_classes).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    
    if not models:
        print("No models loaded!")
        return

    y_true = []
    y_pred = []

    print(f"Evaluating with {len(models)} models...")
    with torch.no_grad():
        for seq, label in test_data:
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            
            # 全モデルの予測確率を加算（Soft Voting）
            avg_logits = torch.zeros(1, num_classes).to(device)
            for model in models:
                _, logits = model(seq_tensor)
                avg_logits += logits
            
            # 平均を取らなくてもargmaxの結果は同じだが、概念的に平均
            avg_logits /= len(models)
            
            pred_label = torch.argmax(avg_logits, dim=1).item()
            y_true.append(label)
            y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Ensemble Evaluation Result ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    names = [id_to_cat[i] for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate()
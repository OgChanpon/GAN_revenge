# evaluate_ensemble_hard.py (Hard Voting版)

import torch
import pickle
import numpy as np
import os
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models import Discriminator

# --- 設定 ---
MODEL_FILES = [f"finetuned_ensemble_full_{i}.pth" for i in range(5)]
TEST_DATA_PATH = 'test_dataset_filtered_full.pkl'
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered_full.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"Loading data...")
    with open(WORD_TO_INT_PATH, 'rb') as f: vocab_size = len(pickle.load(f))
    with open(CATEGORY_TO_ID_PATH, 'rb') as f: 
        cat_to_id = pickle.load(f)
        num_classes = len(cat_to_id)
    with open(TEST_DATA_PATH, 'rb') as f: test_data = pickle.load(f)

    # モデルロード
    models = []
    for path in MODEL_FILES:
        if not os.path.exists(path): continue
        m = Discriminator(vocab_size, 256, 512, num_classes).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)
    
    y_true = []
    y_pred = []

    print(f"Evaluating with {len(models)} models (Hard Voting)...")
    with torch.no_grad():
        for seq, label in test_data:
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            
            # 各モデルの「予測クラスID」を集める
            votes = []
            for model in models:
                _, logits = model(seq_tensor)
                pred = torch.argmax(logits, dim=1).item()
                votes.append(pred)
            
            # 多数決 (最頻値) を取る
            # scipy.stats.mode は (mode_array, count_array) を返す
            final_pred = stats.mode(votes, keepdims=True)[0][0]
            
            y_true.append(label)
            y_pred.append(final_pred)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Ensemble (Hard Voting) Result ===")
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    names = [id_to_cat[i] for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate()
# evaluate_transformer_acgan.py

import torch
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models_transformer_acgan import TransformerDiscriminator # 新モデル

MODEL_PATH = 'transformer_acgan_finetuned.pth'
WORD_TO_INT_PATH = 'word_to_int.pkl'
CATEGORY_TO_ID_PATH = 'category_to_id_filtered.pkl' # フィルタ済み
TEST_DATA_PATH = 'test_dataset_filtered.pkl'        # フィルタ済み

EMBEDDING_DIM = 256
HIDDEN_DIM = 512
MAX_LEN = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    with open(WORD_TO_INT_PATH, 'rb') as f: vocab_size = len(pickle.load(f))
    with open(CATEGORY_TO_ID_PATH, 'rb') as f: 
        cat_to_id = pickle.load(f)
        num_classes = len(cat_to_id)
    with open(TEST_DATA_PATH, 'rb') as f: test_data = pickle.load(f)

    # モデルロード
    model = TransformerDiscriminator(
        vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_classes, 
        num_heads=8, num_layers=2, max_len=MAX_LEN
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    
    print("Evaluating Transformer-AC-GAN...")
    with torch.no_grad():
        for seq, label in test_data:
            seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
            # Dの戻り値は (validity, class_logits)
            _, class_logits = model(seq_tensor)
            pred_label = torch.argmax(class_logits, dim=1).item()
            y_true.append(label)
            y_pred.append(pred_label)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    id_to_cat = {v: k for k, v in cat_to_id.items()}
    names = [id_to_cat[i] for i in range(num_classes)]
    print(classification_report(y_true, y_pred, target_names=names, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    evaluate()
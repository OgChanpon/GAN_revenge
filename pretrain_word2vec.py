# pretrain_word2vec.py
import pickle
import numpy as np
import torch
from gensim.models import Word2Vec
import os

# --- 設定 ---
EMBEDDING_DIM = 256  # models.py と合わせる
WINDOW_SIZE = 5      # 前後5単語を見る
MIN_COUNT = 1        # 1回しか出ない単語も学習する
WORKERS = 4          # 並列数
EPOCHS = 10          # Word2Vecの学習回数

def create_embedding_matrix():
    print("データをロード中...")
    if not os.path.exists("train_dataset.pkl") or not os.path.exists("word_to_int.pkl"):
        print("エラー: データが見つかりません。")
        return

    # データの読み込み
    with open("train_dataset.pkl", "rb") as f:
        train_data = pickle.load(f)
    
    with open("word_to_int.pkl", "rb") as f:
        word_to_int = pickle.load(f)

    vocab_size = len(word_to_int)
    print(f"ボキャブラリサイズ: {vocab_size}")

    # Word2Vecは「文字列のリストのリスト」を入力とするため変換
    # train_data は [(seq, label), ...]
    sequences = [item[0] for item in train_data]
    
    # トークンID(int)を文字列(str)に変換してリスト化
    sentences = [[str(token) for token in seq] for seq in sequences]

    print("Word2Vec学習開始...")
    w2v_model = Word2Vec(
        sentences, 
        vector_size=EMBEDDING_DIM, 
        window=WINDOW_SIZE, 
        min_count=MIN_COUNT, 
        workers=WORKERS,
        epochs=EPOCHS
    )
    
    print("学習完了。Embedding行列を作成中...")
    
    # PyTorchのnn.Embedding用の行列 (Vocab x Dim) を初期化
    # 0番目はパディング用なので0埋め、他はWord2Vecの結果を埋める
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    
    hit_count = 0
    for word, i in word_to_int.items():
        if str(i) in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[str(i)]
            hit_count += 1
            
    print(f"適合率: {hit_count}/{vocab_size} の単語ベクトルを初期化しました。")
    
    # Tensorに変換して保存
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
    torch.save(embedding_tensor, "embedding_matrix.pth")
    print("保存完了: embedding_matrix.pth")

if __name__ == "__main__":
    create_embedding_matrix()
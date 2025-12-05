# tokenka.py (Mal-API-2019 フォルダ構成対応版)

import os
import pandas as pd
import pickle
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from collections import Counter

# --- 設定 ---
if len(sys.argv) < 2:
    print("使い方: python3 tokenka.py <malware_api_classフォルダへのパス>")
    sys.exit()

# 引数で受け取るのは親フォルダ (malware_api_class)
dataset_root = sys.argv[1]

# ★★★ パス指定を実際の構成に合わせました ★★★
# ラベルは直下にある
LABEL_DATA_FILE = os.path.join(dataset_root, "labels.csv")
# データはサブフォルダ (mal-api-2019) の中にある
API_DATA_FILE = os.path.join(dataset_root, "mal-api-2019", "all_analysis_data.txt")

# 出力ファイル名
WORD_TO_INT_FILE = 'word_to_int.pkl'
CATEGORY_TO_ID_FILE = 'category_to_id.pkl'
TRAIN_DATASET_FILE = 'train_dataset.pkl'
TEST_DATASET_FILE = 'test_dataset.pkl'

# パラメータ
TEST_SPLIT_RATIO = 0.1
MIN_LEN = 10
MAX_LEN = 1000

# --- 1. データの読み込み ---
print("Step 1: データファイルを読み込み中...")
print(f"API Data: {API_DATA_FILE}")
print(f"Label Data: {LABEL_DATA_FILE}")

if not os.path.exists(API_DATA_FILE) or not os.path.exists(LABEL_DATA_FILE):
    print(f"エラー: ファイルが見つかりません。パスを確認してください。")
    sys.exit()

try:
    # APIデータ (テキストファイル)
    # ファイルが大きいので少し時間がかかるかもしれません
    with open(API_DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        api_lines = f.readlines()
    
    # ラベルデータ (CSV)
    df_label = pd.read_csv(LABEL_DATA_FILE, header=None, names=['label'])
    labels = df_label['label'].tolist()

except Exception as e:
    print(f"エラー: 読み込み失敗: {e}")
    sys.exit()

if len(api_lines) != len(labels):
    print(f"警告: データ数({len(api_lines)})とラベル数({len(labels)})が一致しません。")
    min_len = min(len(api_lines), len(labels))
    api_lines = api_lines[:min_len]
    labels = labels[:min_len]

print(f"読み込み完了。総データ数: {len(api_lines)} 件")

# --- 2. 前処理とフィルタリング ---
print("\nStep 2: データの前処理とフィルタリング...")

valid_data = []
all_apis = set()
label_counter = Counter()

# プログレス表示用
total_lines = len(api_lines)

for i, (line, label) in enumerate(zip(api_lines, labels)):
    if (i+1) % 1000 == 0:
        print(f"処理中... {i+1}/{total_lines}")

    # APIシーケンスをリストに変換
    api_list = line.strip().split()
    
    # フィルタリング & 切り詰め
    if len(api_list) >= MIN_LEN:
        if len(api_list) > MAX_LEN:
            api_list = api_list[:MAX_LEN]
            
        valid_data.append((api_list, label))
        all_apis.update(api_list)
        label_counter[label] += 1

print(f"フィルタリング完了。有効データ数: {len(valid_data)} 件")
print("\n--- カテゴリ別データ数 ---")
for label, count in label_counter.most_common():
    print(f"{label}: {count}")
print("--------------------------")

# --- 3. 辞書の作成 ---
print("\nStep 3: 辞書を作成中...")

# API辞書
word_to_int = {"_PAD_": 0, "_UNK_": 1}
for i, api in enumerate(sorted(list(all_apis))):
    word_to_int[api] = i + 2

with open(WORD_TO_INT_FILE, "wb") as f:
    pickle.dump(word_to_int, f)
print(f"API語彙数: {len(word_to_int)} -> '{WORD_TO_INT_FILE}'")

# カテゴリ辞書
categories = sorted(list(label_counter.keys()))
category_to_id = {cat: i for i, cat in enumerate(categories)}

with open(CATEGORY_TO_ID_FILE, "wb") as f:
    pickle.dump(category_to_id, f)
print(f"カテゴリ数: {len(category_to_id)} -> '{CATEGORY_TO_ID_FILE}'")
print(f"カテゴリID: {category_to_id}")

# --- 4. トークン化と保存 ---
print("\nStep 4: データを数値変換して保存中...")

all_tokenized_data = []
for api_list, label in valid_data:
    token_ids = [word_to_int.get(api, word_to_int["_UNK_"]) for api in api_list]
    label_id = category_to_id[label]
    all_tokenized_data.append((token_ids, label_id))

# 訓練用とテスト用に分割
train_data, test_data = train_test_split(
    all_tokenized_data, 
    test_size=TEST_SPLIT_RATIO, 
    random_state=42, 
    stratify=[label for _, label in all_tokenized_data]
)

with open(TRAIN_DATASET_FILE, "wb") as f:
    pickle.dump(train_data, f)
print(f"訓練データ: {len(train_data)} 件 -> '{TRAIN_DATASET_FILE}'")

with open(TEST_DATASET_FILE, "wb") as f:
    pickle.dump(test_data, f)
print(f"テストデータ: {len(test_data)} 件 -> '{TEST_DATASET_FILE}'")

print("\n--- 完了しました ---")
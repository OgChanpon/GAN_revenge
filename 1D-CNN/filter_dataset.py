# filter_dataset.py
import pickle
import os

# --- 除外したいクラス名を設定 ---
EXCLUDE_CLASSES = ['Trojan', 'Spyware', 'Dropper']

# ファイルパス
TRAIN_PKL = 'train_dataset.pkl'
TEST_PKL = 'test_dataset.pkl'
CAT_PKL = 'category_to_id.pkl'

NEW_TRAIN_PKL = 'train_dataset_filtered_full.pkl'
NEW_TEST_PKL = 'test_dataset_filtered_full.pkl'
NEW_CAT_PKL = 'category_to_id_filtered_full.pkl'

def filter_and_save():
    if not os.path.exists(TRAIN_PKL) or not os.path.exists(CAT_PKL):
        print("エラー: 元データが見つかりません。")
        return

    print(f"除外対象クラス: {EXCLUDE_CLASSES}")

    # 1. カテゴリ辞書の読み込みと再構築
    with open(CAT_PKL, 'rb') as f:
        old_cat_to_id = pickle.load(f)
    
    # 除外IDを特定
    exclude_ids = []
    new_cat_to_id = {}
    old_id_to_new_id = {}
    
    current_new_id = 0
    
    print("\n--- クラスIDの再割り当て ---")
    for cat, old_id in old_cat_to_id.items():
        if cat in EXCLUDE_CLASSES:
            exclude_ids.append(old_id)
            print(f"[-] 除外: {cat} (Old ID: {old_id})")
        else:
            new_cat_to_id[cat] = current_new_id
            old_id_to_new_id[old_id] = current_new_id
            print(f"[+] 保持: {cat} (Old ID: {old_id} -> New ID: {current_new_id})")
            current_new_id += 1
            
    print(f"\n新しいクラス数: {len(new_cat_to_id)}")

    # 2. データのフィルタリング関数
    def filter_data(file_path, save_path):
        print(f"\n処理中: {file_path} ...")
        with open(file_path, 'rb') as f:
            data = pickle.load(f) # [(seq, label), ...]
            
        original_count = len(data)
        new_data = []
        
        for seq, label in data:
            if label not in exclude_ids:
                # 新しいIDに変換して保存
                new_label = old_id_to_new_id[label]
                new_data.append((seq, new_label))
                
        print(f"  - 元データ数: {original_count}")
        print(f"  - フィルタ後: {len(new_data)}")
        print(f"  - 保存先: {save_path}")
        
        with open(save_path, 'wb') as f:
            pickle.dump(new_data, f)

    # 3. 実行
    filter_data(TRAIN_PKL, NEW_TRAIN_PKL)
    filter_data(TEST_PKL, NEW_TEST_PKL)
    
    # 新しい辞書を保存
    with open(NEW_CAT_PKL, 'wb') as f:
        pickle.dump(new_cat_to_id, f)
    print(f"\n新しいカテゴリ辞書を保存: {NEW_CAT_PKL}")
    print("完了！")

if __name__ == '__main__':
    filter_and_save()
# -*- coding: utf-8 -*-
import os
from huggingface_hub import login, HfApi, create_repo, upload_folder, whoami

# ==== 設定 ====
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # 環境変数があれば使う
REPO_ID = "ayarnte/IRM_high_ver"          # ← 新しいモデル名
LOCAL_DIR = "./irm_deberta_unc_iso"       # ← 学習アウトの保存先（あなたのケース）
PRIVATE = False                           # 公開なら False, 非公開なら True

# （任意）モデルカードを同時に置く場合
MODEL_CARD = f"""---
library_name: transformers
tags:
  - regression
  - creativity
  - iclr
license: mit
pipeline_tag: text-classification
---

# IRM High Ver

DeBERTa-v3-large をベースにした Idea Reward Model（不確実性回帰 + Isotonic 校正）。
入力: タイトル + アブストラクト → 回帰スコア（μ）と 0–1 の報酬にマッピング。
"""

def main():
    # 1) ログイン
    if HF_TOKEN:
        login(HF_TOKEN)
    else:
        # すでに login 済みならそのまま通る
        try:
            _ = whoami()
        except Exception:
            print(">>> 初回は login() が必要です。`from huggingface_hub import login; login()` を先に実行してください。")
            raise

    # 2) リポジトリ作成（存在しても OK）
    try:
        create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)
        print(f"[OK] Repo ready: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print("[WARN] create_repo で警告/エラー:", e)

    # 3) 必要なら README を用意
    readme_path = os.path.join(LOCAL_DIR, "README.md")
    if not os.path.exists(readme_path):
        os.makedirs(LOCAL_DIR, exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(MODEL_CARD)

    # 4) フォルダアップロード
    #   - allow_patterns / ignore_patterns で不要ファイル除外可
    #   - largeファイルは自動で LFS に乗ります
    commit = upload_folder(
        folder_path=LOCAL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=["*.json", "*.jsonl", "*.safetensors", "*.bin", "*.pt", "*.model", "*.txt", "*.md", "*.joblib"],
        # ignore_patterns=["wandb/**", "**/.ipynb_checkpoints/**"],
    )
    print("✅ Upload completed")
    print("  repo:", REPO_ID)
    print("  url :", f"https://huggingface.co/{REPO_ID}")
    print("  commit:", commit.oid)

if __name__ == "__main__":
    main()
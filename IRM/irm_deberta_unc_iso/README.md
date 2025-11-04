---
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

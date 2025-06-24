from datasets import load_dataset
ds = load_dataset("imdb")     # ここはネット必須。完了まで数分
ds.save_to_disk(r"D:/LLMModel/imdb_dataset_arrow")
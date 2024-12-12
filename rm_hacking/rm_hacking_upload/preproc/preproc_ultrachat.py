from datasets import load_dataset

data_dir = "/workspace/jihaozhe/data/ultrachat_200k"

data = load_dataset(data_dir, split="train_sft")

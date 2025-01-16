import base64
from io import BytesIO
from mimetypes import guess_type
from datasets import load_dataset, Dataset, load_from_disk
import json
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = './base_datasets/LLaVA-RLAIF-Data'
data_files = [f'{data_path}/RLAIF-V-Dataset_{i:03d}.parquet' for i in range(14)]
dataset = load_dataset('parquet', data_files=data_files)
print(dataset)
origin_datasets = dataset['train']['origin_dataset']
counts = Counter(origin_datasets)
total_count = len(origin_datasets)
proportions = {ds: count / total_count for ds, count in counts.items()}
print("=====================================")
print("Origin dataset proportions: total count:", total_count)
for ds, proportion in proportions.items():
    print(f"{ds}: {proportion:.2%}")
print("=====================================")
Sample_num = 2500 * 4
df = pd.DataFrame(dataset['train'])

_, df_sampled0 = train_test_split(df, test_size=Sample_num, stratify=df['origin_dataset'], random_state=42)
df_sampled00, df_sampled01 = train_test_split(df_sampled0, test_size=Sample_num//2, stratify=df_sampled0['origin_dataset'], random_state=42)
df_sampled1, df_sampled2 = train_test_split(df_sampled00, test_size=Sample_num//4, stratify=df_sampled00['origin_dataset'], random_state=42)
df_sampled3, df_sampled4 = train_test_split(df_sampled01, test_size=Sample_num//4, stratify=df_sampled01['origin_dataset'], random_state=42)

for df_sampled in (df_sampled1, df_sampled2, df_sampled3, df_sampled4):
    counts = Counter(df_sampled['origin_dataset'])
    total_count = len(df_sampled['origin_dataset'])
    proportions = {ds: count / total_count for ds, count in counts.items()}
    print("=====================================")
    print("Sampled dataset proportions: total count:", total_count)
    for ds, proportion in proportions.items():
        print(f"{ds}: {proportion:.2%}")
    print("=====================================")

save_paths = {
    'df_sampled1': './base_datasets/LLaVA-RLAIF-SubData/subset1',
    'df_sampled2': './base_datasets/LLaVA-RLAIF-SubData/subset2',
    'df_sampled3': './base_datasets/LLaVA-RLAIF-SubData/subset3',
    'df_sampled4': './base_datasets/LLaVA-RLAIF-SubData/subset4',
}

for name, df in zip(save_paths.keys(), [df_sampled1, df_sampled2, df_sampled3, df_sampled4]):
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(save_paths[name])
# Load the dataset from disk
A = load_from_disk(save_paths['df_sampled1'])
dataset = {"train": A}

import json
import os
import re
import shutil

def sort_key(file_name):
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]), int(numbers[1])

def load_json_files(json_dir):
    all_data = []
    for file_name in sorted(os.listdir(json_dir), key=sort_key):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
    return all_data

def has_repeating_last_sentence(report):
    sentences = report.split('.')
    if len(sentences) < 2:
        return False
    last_sentence = sentences[-2].strip()
    report_without_last_sentence = '.'.join(sentences[:-2])
    if last_sentence in report_without_last_sentence:
        print(report)
        return True
    else:
        return False

def has_repeating_last_word(report):
    words = report.split()
    if len(words) < 2:
        return False
    last_word = words[-1].strip()
    if words[:-2].count(last_word) > 30:
        print(report)
        return True

def count_words(text):
    words = text.split()
    return len(words)

# TODO: For LLaVA-1.5-7B Model
json_dir = ['./output/llava7b_online_generation_subset1/rollouts',
            './output/llava7b_online_generation_subset2/rollouts',
            './output/llava7b_online_generation_subset3/rollouts',
            './output/llava7b_online_generation_subset4/rollouts',]
OPA_datapath = "./base_datasets/opa_training_data-7B"
OPA_DPO_datapath = "./base_datasets/opadpo_training_data-7B"

# # TODO: For LLaVA-1.5-13B Model
# json_dir = ['./output/llava13b_online_generation_subset1/rollouts',
#             './output/llava13b_online_generation_subset2/rollouts',
#             './output/llava13b_online_generation_subset3/rollouts',
#             './output/llava13b_online_generation_subset4/rollouts',]
# OPA_datapath = "./base_datasets/opa_training_data-13B"
# OPA_DPO_datapath = "./base_datasets/opadpo_training_data-13B"


if isinstance(json_dir, list):
    mixed_data = True
    merged_data = []
    for dir in json_dir:
        if os.path.exists(dir):
            merged_data.extend(load_json_files(dir))
        else:
            print(f"Directory {dir} does not exist.")
else:
    mixed_data = False
    merged_data = load_json_files(json_dir)
original_size = len(merged_data)

count = 0
image_id_cache = []
for item in merged_data:
    if item['image_id'] not in image_id_cache:
        image_id_cache.append(item['image_id'])
    item['AI_json_report'] = json.dumps(item['AI_json_report'], ensure_ascii=False, indent=4)
    if 'USER:  \n' in item['query']:
        item['query'] = item['query'][item['query'].find('USER:  \n') + 8:]
        item['query'] = '<image>\n' + item['query']
print(f"Number of unique image_id: {len(image_id_cache)}")

filtered_data = [item for item in merged_data if item['AI_json_report'] != '""']
filtered_size = len(filtered_data)
print("Filter1: filtered out empty AI_json_report")
print(f"Original size: {original_size}, Filtered size: {filtered_size}")

filtered_data = [item for item in filtered_data if not has_repeating_last_sentence(item['original_generate_response']) and not has_repeating_last_word(item['original_generate_response'])]
filtered_size2 = len(filtered_data)
print("Filter2: filtered out repeating last sentence")
print(f"Filtered size: {filtered_size}, Filtered size2: {filtered_size2}")

filtered_data = [item for item in filtered_data if isinstance(item.get('AI_pseudo_response', ''), str) and len(item.get('AI_pseudo_response', '')) > 0]
print("Filter3: filtered out empty AI_pseudo_response")
print(f"Filtered size2: {filtered_size2}, Filtered size3: {len(filtered_data)}")

from datasets import Dataset

# NOTE: Saving OPA-Dataset
if os.path.exists(OPA_datapath):
    print(f"Removing existing file: {OPA_datapath}")
    shutil.rmtree(OPA_datapath)

opa_parent_dir = os.path.dirname(OPA_datapath)
if not os.path.exists(opa_parent_dir):
    print(f"Creating directory: {opa_parent_dir}")
    os.makedirs(opa_parent_dir)

dataset = Dataset.from_dict({
    "queries": [item['query'] for item in filtered_data],
    "image_bytes": [item['image_bytes'] for item in filtered_data],
    "standard_response": [item['standard_response'] for item in filtered_data],
    "AI_pseudo_response": [item['AI_pseudo_response'] for item in filtered_data],
})
dataset.save_to_disk(OPA_datapath)

# * DPO-Dataset
dataset = Dataset.from_dict({
    "queries": [item['query'] for item in filtered_data],
    "image_bytes": [item['image_bytes'] for item in filtered_data],
    "standard_response": [item['standard_response'] for item in filtered_data],
    "original_generate_response": [item['original_generate_response'] for item in filtered_data],
    "AI_pseudo_response": [item['AI_pseudo_response'] for item in filtered_data],
    "AI_json_report": [item['AI_json_report'] for item in filtered_data],
})
dataset.save_to_disk(OPA_DPO_datapath)

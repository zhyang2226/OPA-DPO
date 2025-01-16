import copy
import base64
import io
import os
import json
import difflib

from dataclasses import dataclass
from typing import Dict, Sequence, Union
from PIL import Image
from pathlib import Path

import torch
import transformers
# from datasets import load_dataset
# from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
# from datasets import load_from_disk
# from data_utils.image_loading import load_image

from utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from typing import Sequence, Dict, List
from transformers import AutoTokenizer, AutoProcessor, DataCollatorWithPadding

score_mapping = {
    1: 2.5,
    2: 2.0,
    3: 1.5,
    4: 1.0,
}
error_type_mapping = {
    'image_recognition_error': 3.0,
    'correct': 1.0,
    'language_comprehension_error': 1.0,
}

def pad_and_stack(tensor_list, pad_value, max_length=None):
    if max_length is None:
        max_length = max(tensor.size(0) for tensor in tensor_list)
    padded_tensors = []
    for tensor in tensor_list:
        padding_length = max_length - tensor.size(0)
        padding_tensor = torch.full((padding_length,), pad_value, dtype=tensor.dtype)
        padded_tensor = torch.cat([tensor, padding_tensor])
        padded_tensors.append(padded_tensor)
    stacked_tensor = torch.stack(padded_tensors)
    return stacked_tensor

def complete_copied_content(original_string, string_list):
    fixed_list = []
    remaining_string = original_string
    for s in string_list:
        trimmed_s = s.strip()
        if len(trimmed_s) > 0:
            pos = remaining_string.find(trimmed_s)
            if pos != -1:
                fixed_list.append(remaining_string[:pos + len(trimmed_s)].strip(' '))
                remaining_string = remaining_string[pos + len(trimmed_s):]
            else:
                # print("String Matching Failed")
                return string_list
        else:
            fixed_list.append("")
    if len(fixed_list) > 0 and remaining_string.strip():
        fixed_list[-1] += remaining_string.strip()
    return fixed_list

def add_eos(tensor: Union[torch.Tensor, Dict], pad_id: int, eos_id: int) -> Union[torch.Tensor, Dict]:
    if hasattr(tensor, 'data'):
        tensor = tensor.data
    if isinstance(tensor, Dict):
        for key, value in tensor.items():
            tensor[key] = add_eos(value, pad_id, eos_id)
    elif isinstance(tensor, torch.Tensor):
        for row in tensor:
            pad_index = (row == pad_id).nonzero(as_tuple=True)[0]
            if pad_index.numel() > 0:
                row[pad_index[0]] = eos_id
    else:
        raise ValueError("Unsupported type for `tensor`")
    return tensor

def pad_eos(response_tensor, score_tensor, eos_id: int) -> torch.Tensor:
    if isinstance(response_tensor, torch.Tensor) and isinstance(score_tensor, torch.Tensor):
        for row_response, row_score in zip(response_tensor, score_tensor):
            pad_index = (row_response == eos_id).nonzero(as_tuple=True)[0]
            if pad_index.numel() > 0:
                pad_value = row_score[pad_index[0] - 1] if row_score[pad_index[0] - 1] != 0 else 1
                row_score[pad_index[0]] = pad_value
    else:
        raise ValueError("Unsupported type for `tensor`")
    return score_tensor

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: AutoTokenizer
    query_len: 128
    response_len: 896
    detailed_report: False

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # Separate the instances into individual components
        queries = [instance['queries'] for instance in instances]
        images = [instance['images'] for instance in instances]
        standard_responses = [instance['standard_response'] for instance in instances]
        original_generate_responses = [instance['original_generate_response'] for instance in instances]
        AI_pseudo_responses = [instance['AI_pseudo_response'] for instance in instances]
        if self.detailed_report:
            AI_json_reports = [json.loads(instance['AI_json_report']) for instance in instances]

        # Tokenize the text fields with padding and specified max lengths
        self.tokenizer.padding_side = 'left'
        queries_encodings = self.tokenizer(queries, padding='max_length', truncation=True, max_length=self.query_len, return_tensors='pt')
        queries_encodings['input_ids'][queries_encodings['input_ids']==30861] = IMAGE_TOKEN_INDEX
        self.tokenizer.padding_side = 'right'
        standard_response_encodings = self.tokenizer(standard_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
        standard_response_encodings = add_eos(standard_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
        if not self.detailed_report:
            original_generate_response_encodings = self.tokenizer(original_generate_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
            original_generate_response_encodings = add_eos(original_generate_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
            AI_pseudo_response_encodings = self.tokenizer(AI_pseudo_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
            AI_pseudo_response_encodings = add_eos(AI_pseudo_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
            batch = {
                'queries': queries_encodings['input_ids'],
                'queries_attention_mask': queries_encodings['attention_mask'],
                'standard_response': standard_response_encodings['input_ids'],
                'standard_response_attention_mask': standard_response_encodings['attention_mask'],
                'original_generate_response': original_generate_response_encodings['input_ids'],
                'original_generate_response_attention_mask': original_generate_response_encodings['attention_mask'],
                'AI_pseudo_response': AI_pseudo_response_encodings['input_ids'],
                'AI_pseudo_response_attention_mask': AI_pseudo_response_encodings['attention_mask'],
            }
        else:
            try:
                # ! Delete image_description from AI_json_reports
                AI_json_reports = [{key: value for key, value in report.items() if key not in ['image_description', 'image description']} for report in AI_json_reports]
                # TODO: Bug exist that the "original_generate_responses" and "copied_content" are not aligned in the AI_json_reports
                for i, report in enumerate(AI_json_reports):
                    orig = original_generate_responses[i]
                    copied_list = []
                    for key in report.keys():
                        copied_content = report[key].get('copied content', report[key].get('copied_content', ''))
                        if copied_content:  # Check if copied content is not empty
                            copied_list.append(copied_content)
                        else:
                            copied_list.append('')  # Append empty string if no copied content
                    completed_copied_list = complete_copied_content(orig, copied_list)
                    for j, key in enumerate(report.keys()):
                        if completed_copied_list[j]:  # Only update if completed content is not empty
                            if 'copied content' in report[key]:
                                report[key]['copied content'] = completed_copied_list[j]
                            elif 'copied_content' in report[key]:
                                report[key]['copied_content'] = completed_copied_list[j]

                original_generate_response_encodings = []
                AI_pseudo_response_encodings = []
                original_generate_response_scores = []
                AI_pseudo_response_scores = []
                original_generate_response_image_relations = []
                AI_pseudo_response_image_relations = []
                for AI_json_report in AI_json_reports:
                    original_generate_response_encoding = []
                    AI_pseudo_response_encoding = []
                    original_generate_response_score = []
                    AI_pseudo_response_score = []
                    original_generate_response_image_relation = []
                    AI_pseudo_response_image_relation = []
                    count = 0
                    for key in AI_json_report.keys():
                        copied_content = None
                        rewritten_content = self.tokenizer(AI_json_report[key].get('rewritten content', AI_json_report[key].get('rewritten_content', '')), return_tensors='pt')['input_ids']
                        if rewritten_content.size(1) <= 1:  # Check if rewritten_content is an empty string
                            rewritten_content = None
                        else:
                            rewritten_content = rewritten_content[:, 1:] if count != 0 else rewritten_content

                        if key != 'Added':
                            copied_content = self.tokenizer(AI_json_report[key].get('copied content', AI_json_report[key].get('copied_content', '')), return_tensors='pt')['input_ids']
                            if copied_content.size(1) <= 1:  # Check if copied_content is an empty string
                                copied_content = None
                            else:
                                copied_content = copied_content[:, 1:] if count != 0 else copied_content
                                if copied_content[:, 0] == torch.tensor(29871): # Check if the first token is empty string ''
                                    copied_content = copied_content[:, 1:]

                            score = AI_json_report[key].get('score', 4)  # Default to 4 if 'score' is not present
                            error_type = AI_json_report[key].get('error type', AI_json_report[key].get('error_type', 'correct'))  # Default to 'correct' if neither 'error type' nor 'error_type' is present
                            copied_content_score = torch.ones_like(copied_content) * score_mapping.get(score, 1.0) if copied_content is not None else None
                            rewritten_content_score = torch.ones_like(rewritten_content) * score_mapping.get(score, 1.0) if rewritten_content is not None else None
                            copied_image_score = torch.ones_like(copied_content) * error_type_mapping.get(error_type, 1.0) if copied_content is not None else None
                            rewritten_image_score = torch.ones_like(rewritten_content) * error_type_mapping.get(error_type, 1.0) if rewritten_content is not None else None
                        else:
                            rewritten_content_score = torch.ones_like(rewritten_content) if rewritten_content is not None else None
                            rewritten_image_score = torch.ones_like(rewritten_content) if rewritten_content is not None else None

                        if copied_content is not None:
                            original_generate_response_encoding.append(copied_content)
                            original_generate_response_score.append(copied_content_score)
                            original_generate_response_image_relation.append(copied_image_score)

                        if rewritten_content is not None:
                            AI_pseudo_response_encoding.append(rewritten_content)
                            AI_pseudo_response_score.append(rewritten_content_score)
                            AI_pseudo_response_image_relation.append(rewritten_image_score)
                        count += 1

                    original_generate_response_encodings.append(torch.cat(original_generate_response_encoding, dim=1)[0])
                    AI_pseudo_response_encodings.append(torch.cat(AI_pseudo_response_encoding, dim=1)[0])
                    original_generate_response_scores.append(torch.cat(original_generate_response_score, dim=1)[0])
                    AI_pseudo_response_scores.append(torch.cat(AI_pseudo_response_score, dim=1)[0])
                    original_generate_response_image_relations.append(torch.cat(original_generate_response_image_relation, dim=1)[0])
                    AI_pseudo_response_image_relations.append(torch.cat(AI_pseudo_response_image_relation, dim=1)[0])

                original_generate_response_encodings = pad_and_stack(original_generate_response_encodings, self.tokenizer.pad_token_id, max_length=self.response_len)
                original_generate_response_encodings = add_eos(original_generate_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
                AI_pseudo_response_encodings = pad_and_stack(AI_pseudo_response_encodings, self.tokenizer.pad_token_id, max_length=self.response_len)
                AI_pseudo_response_encodings = add_eos(AI_pseudo_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)

                # NOTE THAT We DONOT ADD EOS TOKEN TO ORIGINAL GENERATE RESPONSES
                original_generate_response_scores = pad_and_stack(original_generate_response_scores, 0.0, max_length=self.response_len)
                AI_pseudo_response_scores = pad_and_stack(AI_pseudo_response_scores, 0.0, max_length=self.response_len)
                AI_pseudo_response_scores = pad_eos(AI_pseudo_response_encodings, AI_pseudo_response_scores, self.tokenizer.eos_token_id)
                # NOTE THAT We DONOT ADD EOS TOKEN TO ORIGINAL GENERATE RESPONSES
                original_generate_response_image_relations = pad_and_stack(original_generate_response_image_relations, 0.0, max_length=self.response_len)
                AI_pseudo_response_image_relations = pad_and_stack(AI_pseudo_response_image_relations, 0.0, max_length=self.response_len)
                AI_pseudo_response_image_relations = pad_eos(AI_pseudo_response_encodings, AI_pseudo_response_image_relations, self.tokenizer.eos_token_id)


                # original_generate_response_encodings1 = self.tokenizer(original_generate_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
                # original_generate_response_encodings1 = add_eos(original_generate_response_encodings1, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
                # AI_pseudo_response_encodings1 = self.tokenizer(AI_pseudo_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
                # AI_pseudo_response_encodings1 = add_eos(AI_pseudo_response_encodings1, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
                # print(f"Generation Diff: {torch.sum(original_generate_response_encodings1['input_ids'] != original_generate_response_encodings, dim=1)}")
                # print(f"AI_Pseudo  Diff: {torch.sum(AI_pseudo_response_encodings1['input_ids'] != AI_pseudo_response_encodings, dim=1)}")
                # print("==" * 40)

                # Create the batch dictionary
                batch = {
                    'queries': queries_encodings['input_ids'],
                    'queries_attention_mask': queries_encodings['attention_mask'],
                    'standard_response': standard_response_encodings['input_ids'],
                    'standard_response_attention_mask': standard_response_encodings['attention_mask'],
                    'original_generate_response': original_generate_response_encodings,
                    'original_generate_response_attention_mask': original_generate_response_encodings!=self.tokenizer.pad_token_id,
                    'AI_pseudo_response': AI_pseudo_response_encodings,
                    'AI_pseudo_response_attention_mask': AI_pseudo_response_encodings!=self.tokenizer.pad_token_id,
                    'original_generate_response_scores': original_generate_response_scores,
                    'AI_pseudo_response_scores': AI_pseudo_response_scores,
                    'original_generate_response_image_relations': original_generate_response_image_relations,
                    'AI_pseudo_response_image_relations': AI_pseudo_response_image_relations,
                }
            except Exception as e:
                print(e)
                original_generate_response_encodings = self.tokenizer(original_generate_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
                original_generate_response_encodings = add_eos(original_generate_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
                AI_pseudo_response_encodings = self.tokenizer(AI_pseudo_responses, padding='max_length', truncation=True, max_length=self.response_len, return_tensors='pt')
                AI_pseudo_response_encodings = add_eos(AI_pseudo_response_encodings, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)
                batch = {
                    'queries': queries_encodings['input_ids'],
                    'queries_attention_mask': queries_encodings['attention_mask'],
                    'standard_response': standard_response_encodings['input_ids'],
                    'standard_response_attention_mask': standard_response_encodings['attention_mask'],
                    'original_generate_response': original_generate_response_encodings['input_ids'],
                    'original_generate_response_attention_mask': original_generate_response_encodings['attention_mask'],
                    'AI_pseudo_response': AI_pseudo_response_encodings['input_ids'],
                    'AI_pseudo_response_attention_mask': AI_pseudo_response_encodings['attention_mask'],
                    'original_generate_response_scores': torch.zeros_like(original_generate_response_encodings['input_ids']),
                    'AI_pseudo_response_scores': torch.zeros_like(AI_pseudo_response_encodings['input_ids']),
                    'original_generate_response_image_relations': torch.zeros_like(original_generate_response_encodings['input_ids']),
                    'AI_pseudo_response_image_relations': torch.zeros_like(AI_pseudo_response_encodings['input_ids']),
                }

        if all(x is not None and x.shape == images[0].shape for x in images):
            batch["images"] = torch.stack(images)
        else:
            batch["images"] = images

        return batch

class DPO_Dataset(Dataset):
    def __init__(self, data_args, dataset):
        self.data_args = data_args
        self.processor = data_args.image_processor
        self.dataset = dataset
        self.query_templete_prefix = "<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
        self.query_templete_suffix = " ASSISTANT: "

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load and process image
        # if 'images' in item:
        #     image = load_image(Path(f"{os.environ['IMAGE_DIR']}/{item['images']}"))  # TODO: Original BUG WHY WOKRS
        if 'images' in item:
            image_file = item["images"]
            image_folder = os.environ['IMAGE_DIR']
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        elif 'image_id' in item:
            image_file = item["image_id"]
            image_folder = os.environ['IMAGE_DIR']
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        elif 'image_bytes' in item:
            image_bytes = item["image_bytes"]
            image_bytes = base64.b64decode(image_bytes.encode('utf-8'))
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise ValueError("No image found in the dataset")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(
                        pil_img.mode, (width, width), background_color
                    )
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(
                        pil_img.mode, (height, height), background_color
                    )
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(
                image, tuple(int(x * 255) for x in self.processor.image_mean)
            )
            image = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = self.processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return {
            'queries': self.query_templete_prefix + item['queries'].replace('<image>', '图 ') + self.query_templete_suffix,
            'images': image,
            'standard_response': item['standard_response'],
            'original_generate_response': item['original_generate_response'],
            'AI_pseudo_response': item['AI_pseudo_response'],
            'AI_json_report': item['AI_json_report'],
        }


def make_dpo_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    data_args,
) -> Dict:
    # Load dataset.
    from datasets import load_from_disk
    # data_dir = os.environ['DATA_DIR']
    dataset = load_from_disk(data_args.data_path)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,query_len=128, response_len=896, detailed_report=args.detailed_report,
    )
    return dict(
        train_dataset=DPO_Dataset(data_args, dataset),
        eval_dataset=None,
        data_collator=data_collator,
    )

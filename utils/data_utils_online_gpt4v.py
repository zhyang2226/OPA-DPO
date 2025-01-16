import dataclasses
from typing import Callable, Dict, Optional, List, Sequence

import logging
import pandas as pd

import torch
from torch.utils.data import Dataset

import transformers
import datasets

import utils.common_utils as utils
from utils.common_utils import preprocess, preprocess_multimodal

from PIL import Image
import copy
import os
import tqdm
import io
import base64


logger = logging.getLogger(__name__)

def form_conversation(questions, answers):
    conversation = []
    conversation.append({"from": "human", "value": "<image>\n" + questions})
    conversation.append({"from": "gpt", "value": answers})
    return conversation

def bytes_to_data_url(image_bytes):
    base64_encoded_data = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_encoded_data}"

class QueryResponseDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
        query_len: int,
        df_postprocessor: Optional[Callable] = None,
        data_args: Optional[Dict] = None,
    ):
        self.data_args = data_args
        super(QueryResponseDataset, self).__init__()

        if df_postprocessor is not None:
            df = df_postprocessor(df)
        list_dict_data = df.to_dict(orient="records")

        _s = copy.deepcopy([form_conversation(item['question'], item['chosen']) for item in list_dict_data])
        _s = preprocess_multimodal(_s, data_args)
        _s_target = []

        for __s in _s:
            assert __s[-1]["from"] == "gpt", f"{__s}"
            _s_target.append(__s[-1]["value"])
            __s[-1]["value"] = "\n"

        queries = [
            preprocess(
                [__s],
                tokenizer,
                has_image=True,
                mask_target=False,
                query_len=query_len,
            )["input_ids"]
            for __s in tqdm.tqdm(_s)
        ]

        queries = [
            torch.tensor(query, dtype=torch.long).view(-1)[:-3] for query in queries
        ]

        standard_responses = [
            tokenizer.encode(target, return_tensors='pt').view(-1)[1:] for target in _s_target
        ]

        filtered_queries = []
        filtered_responses = []

        for (query, response) in zip(queries, standard_responses):
            if len(query) <= query_len:
                filtered_queries.append(query)
                filtered_responses.append(response)

        max_query_len = max(len(query) for query in filtered_queries)
        logger.warning(f"Max query length: {max_query_len}")
        max_response_len = max(len(response) for response in filtered_responses)
        logger.warning(f"Max response length: {max_response_len}")

        logger.warning(
            f"Filtered out {len(queries) - len(filtered_queries)} instances out of {len(queries)} that "
            f"exceed length limit. These examples are not used for training, but will still be used in evaluation. "
        )

        queries = torch.stack(
            [
                utils.left_pad(
                    query, target_size=(query_len,), value=tokenizer.pad_token_id
                )
                for query in filtered_queries
            ]
        )
        filtered_responses = [torch.cat([response, torch.tensor([tokenizer.eos_token_id])]) for response in filtered_responses]
        standard_responses = torch.stack(
            [
                utils.right_pad(
                    response, target_size=(max_response_len,), value=tokenizer.pad_token_id
                )
                for response in filtered_responses
            ]
        )

        self.queries = queries
        self.query_attn_masks = queries.ne(tokenizer.pad_token_id).long()
        self.standard_responses = standard_responses

        # Auxiliary data.
        self.list_dict_data = list_dict_data

    def __getitem__(self, idx):
        return_dict = dict(
            queries=self.queries[idx],
            query_attn_masks=self.query_attn_masks[idx],
            standard_responses=self.standard_responses[idx],
        )

        image_bytes = self.list_dict_data[idx]["image"]['bytes']
        image_file = io.BytesIO(image_bytes)
        image_path = self.list_dict_data[idx]["image"]['path']
        image_url = bytes_to_data_url(image_bytes)
        processor = self.data_args.image_processor

        try:
            image = Image.open(image_file).convert("RGB")
        except:
            raise ValueError(f"Error loading image {image_file} for index {idx}")

        if self.data_args.image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return_dict["images"] = image
        return_dict["images_path"] = image_path
        return_dict["images_url"] = image_url
        return_dict["images_bytes"] = image_bytes

        return return_dict

    def __len__(self):
        return len(self.queries)


@dataclasses.dataclass
class DataCollatorForQueryResponseDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        collated_batch = {}
        for key in instances[0].keys():
            if isinstance(instances[0][key], torch.Tensor):
                collated_batch[key] = torch.stack([instance[key] for instance in instances])
            else:
                collated_batch[key] = [instance[key] for instance in instances]
        return collated_batch


def make_rlaif_gpt4v_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
    sample_size: Optional[int] = None,
    batch_idx: int = 0,
    seed: int = 42,
):

    train_instructions = datasets.load_from_disk(data_args.data_path)
    train_df = pd.DataFrame(train_instructions)

    # # Shuffle the DataFrame and reset index
    # train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    # # Select the desired slice of the dataset
    # start_index = batch_idx * sample_size
    # end_index = (batch_idx + 1) * sample_size
    # train_df = train_df.iloc[start_index:end_index].reset_index(drop=True)

    train_dataset = QueryResponseDataset(
        df=train_df,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
        data_args=data_args,
    )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForQueryResponseDataset(),
    )

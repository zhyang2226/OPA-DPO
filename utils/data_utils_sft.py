import copy
import os
import io
import base64

from dataclasses import dataclass
from typing import Dict, Sequence
from PIL import Image

import torch
import transformers
from datasets import load_dataset
from datasets import Dataset

from utils.constants import IGNORE_INDEX

from utils.common_utils import preprocess


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str, pad_token: int):
        input_ids = [instance[key] for instance in instances]  # Flatten.
        try:
            input_ids = pad_sequence_from_left(
                input_ids,
                batch_first=True,
                padding_value=pad_token,
            )
        except:
            raise ValueError(f"Error padding {key} for {input_ids}")
        return input_ids

    def _right_pad_helper(self, instances: Sequence[dict], key: str, pad_token: int):
        input_ids = [instance[key] for instance in instances]  # Flatten.
        try:
            input_ids = pad_sequence_from_right(
                input_ids,
                batch_first=True,
                padding_value=pad_token,
            )
        except:
            raise ValueError(f"Error padding {key} for {input_ids}")
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = self._right_pad_helper(
            instances, "input_ids", self.tokenizer.pad_token_id
        )
        labels = self._right_pad_helper(instances, "labels", IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        assert input_ids.shape == labels.shape
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images

        return batch


class SupervisedVisionLanguageDataset(Dataset):
    def __init__(
        self,
        data_args: Dict,
        hf_dataset: Dataset,
    ):
        super(SupervisedVisionLanguageDataset).__init__()
        self.data_args = data_args
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitems__(self, idx):
        if isinstance(idx, int):
            return self._get_single_item(idx)
        elif isinstance(idx, list):
            return self._get_batch_items(idx)
        else:
            raise ValueError(f"Unsupported index type: {type(idx)}")

    def _get_single_item(self, idx):
        sources = self.hf_dataset[idx]

        image = None
        if "image_id" in sources:
            image_file = sources["image_id"]
            image_folder = self.data_args.image_folder
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert(
                    "RGB"
                )
            except:
                raise ValueError(f"Error loading image {image_file} for index {idx}")
        elif "image_bytes" in sources:
            image_bytes = sources["image_bytes"]
            image_bytes = base64.b64decode(image_bytes.encode('utf-8'))
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            raise NotImplementedError

        processor = self.data_args.image_processor
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
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

        data_dict = copy.deepcopy(sources)
        data_dict = dict(
            input_ids=torch.Tensor(data_dict["input_ids"][0]).long(),
            labels=torch.Tensor(data_dict["labels"][0]).long(),
        )

        if image is not None:
            data_dict["image"] = image

        return data_dict

    def _get_batch_items(self, indices):
        batch = [self._get_single_item(idx) for idx in indices]
        return batch


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


def pad_sequence_from_right(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    return padded_sequence


def extract_v1_dataset(
    example,
    tokenizer,
    data_args,
    has_image=True,
    mask_target=True,
    query_len=None,
    response_len=None,
    response_type=None,
):

    query = example["queries"]
    if response_type == "standard":
        response = example["standard_response"]
    elif response_type == "AI_pseudo":
        response = example["AI_pseudo_response"]
    else:
        raise ValueError(f"Unsupported response type: {response_type}")

    sample = [{'from': 'human', 'value': query}, {'from': 'gpt', 'value': response}]
    sample = preprocess(
        [sample],
        tokenizer,
        has_image=has_image,
        mask_target=mask_target,
        query_len=query_len,
        response_len=response_len,
    )
    return sample


def make_sft_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    data_args,
) -> Dict:
    
    def format_dataset(dataset, dataset_format):
        if dataset_format == "v1":
            dataset1 = dataset.map(
                lambda ex: extract_v1_dataset(
                    ex,
                    tokenizer=tokenizer,
                    data_args=data_args,
                    query_len=128,    
                    response_len=896,
                    response_type="standard",
                ),
                num_proc=16,
            )
            dataset2 = dataset.map(
                lambda ex: extract_v1_dataset(
                    ex,
                    tokenizer=tokenizer,
                    data_args=data_args,
                    query_len=128,
                    response_len=896,
                    response_type="AI_pseudo",
                ),
                num_proc=16,
            )
            from datasets import concatenate_datasets
            dataset = concatenate_datasets([dataset1, dataset2])
        else:
            raise NotImplementedError

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names
                if col not in ["image_id", "input_ids", "labels", "image_bytes", "image"]
            ]
        )
        return dataset

    # Load dataset.
    dataset = Dataset.load_from_disk(args.data_dir)
    dataset = format_dataset(dataset, "v1")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=128, seed=args.seed)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`"
            )
            eval_dataset = dataset["test"]
            
        if args.group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )
    train_dataset = dataset["train"]
    if args.group_by_length:
        train_dataset = train_dataset.map(
            lambda x: {"length": len(x["input"]) + len(x["output"])}
        )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
    )
    return dict(
        train_dataset=SupervisedVisionLanguageDataset(data_args, train_dataset),
        eval_dataset=SupervisedVisionLanguageDataset(data_args, eval_dataset),
        data_collator=data_collator,
    )

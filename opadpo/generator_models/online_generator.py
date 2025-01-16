import base64
import json
import gc
import glob
from itertools import chain
import logging
import os
import pathlib
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import tqdm
import transformers
from loguru import logger as lg

import copy

from utils.constants import (
    IMAGE_TOKEN_INDEX,
)

from utils.data_utils_online_gpt4v import QueryResponseDataset
import utils.common_utils as common_utils
from opadpo.generator_models.generator import (
    AlpacaAccelerator,
    Generator,
    truncate_after_eos_with_padding,
)

from opadpo.generator_models.prompt.pseudo_rollout_coco import (PROMPT_LONG_coco_4V,input_format_coco_4V, output_format_coco_4V)

AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyPathOrNone = Optional[AnyPath]

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[36m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler

# Name of the files used for checkpointing
ADAPTER_MODEL_DIR = "adapter_model"
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
VALUE_HEAD_NAME = "value_head.pt"
SCALER_NAME = "scaler.pt"


class Online_Generator(Generator):
    def __init__(
        self,
        args,
        train_dataset: QueryResponseDataset,
        eval_dataset: QueryResponseDataset,
        data_collator: Callable,
        policy: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: AlpacaAccelerator,
    ):
        super(Online_Generator, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            tokenizer=tokenizer,
            accelerator=accelerator,
        )

    def slice_input_shift_pad(self, input_ids, output_slices, query_len=128, pad_side='left'):
        num_rows, max_len = input_ids.shape
        slices_id = [output_slices[i].start for i in range(num_rows)]
        slices_id_np = np.array(slices_id)

        if slices_id_np.max() > query_len and query_len > 0:
            lg.warning(f"In this batch, there exists at least one query which has length larger than {query_len}")

        if query_len > 0:
            queries = torch.zeros(num_rows, query_len, dtype=input_ids.dtype, device=input_ids.device)
        else:
            queries = torch.zeros(num_rows, slices_id_np.max(), dtype=input_ids.dtype, device=input_ids.device)
        standard_responses = torch.zeros(num_rows, max_len-slices_id_np.min(), dtype=input_ids.dtype, device=input_ids.device)
        query_attn_masks = torch.zeros_like(queries)

        if pad_side == 'left':
            for i in range(num_rows):
                length = slices_id[i]
                rest_length = max_len - length
                # QUERY DOES NOT EXCEED LIMITS / NO LIMITS
                if (length <= query_len and query_len > 0) or query_len<=0:
                    queries[i, -length:] = input_ids[i, :length]
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, -length:] = 1
                # QUERY EXCEEDS LIMITS
                else:
                    print(f"QUERY: {input_ids[i, :length].cpu().numpy()}")
                    queries[i, :] = input_ids[i, :query_len]
                    if IMAGE_TOKEN_INDEX not in input_ids[i, :query_len-9]:
                        lg.warning(f"NO IMAGE TOKEN FOUND in {i}th row")
                        queries[i, -9:] = torch.tensor([IMAGE_TOKEN_INDEX, 29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    else:
                        queries[i, -8:] = torch.tensor([29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, :] = 1
        elif pad_side == 'right':
            for i in range(num_rows):
                length = slices_id[i]
                rest_length = max_len - length
                # QUERY DOES NOT EXCEED LIMITS / NO LIMITS
                if (length <= query_len and query_len > 0) or query_len<=0:
                    queries[i, :length] = input_ids[i, :length]
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, :length] = 1
                # QUERY EXCEEDS LIMITS
                else:
                    queries[i, :] = input_ids[i, :query_len]
                    if IMAGE_TOKEN_INDEX not in input_ids[i, :query_len-9]:
                        lg.warning(f"NO IMAGE TOKEN FOUND in {i}th row")
                        queries[i, -9:] = torch.tensor([IMAGE_TOKEN_INDEX, 29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    else:
                        queries[i, -8:] = torch.tensor([29889, 29871, 319, 1799, 9047, 13566, 29901, 29871], device=input_ids.device)
                    standard_responses[i, :rest_length] = input_ids[i, length:]
                    query_attn_masks[i, :] = 1
        else:
            raise NotImplementedError(f"Note that pad_side '{pad_side}' can only be chosen from 'left' and 'right'!")

        return queries, query_attn_masks, standard_responses

    def process_responses_coco_gpt4v(self, responses, images_url, text_queries, text_responses, text_standard_responses, static_messages, compute_type):
        Pseudo_response = []
        Generated_response = []
        Pseudo_response_ids = []
        Generated_response_ids = []
        Pseudo_rewards = []
        Original_rewards = []
        report_json = []

        for i in range(len(text_responses)):
            try:
                current_messages = static_messages.copy()
                user_content = {
                    "queries": text_queries[i],
                    "generated_response": text_responses[i],
                    "standard_response": text_standard_responses[i],
                }
                current_messages.append(
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": json.dumps(user_content)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": images_url[i]
                            }
                        }
                    ]})
                Report = self.openai_API.call_llm_with_messages(current_messages, model=self.api_model)
                Report = Report.replace("```json", "").replace("```", "")
                Report = json.loads(Report)

                cache_pseudo_response = []
                cache_generated_response = []

                for key in Report.keys():
                    if key == 'image description' or key == 'image_description':
                        pass
                    elif key != 'Added':
                        cache_pseudo_response.append(Report[key].get("rewritten content", Report[key].get("rewritten_content", "")))
                        cache_generated_response.append(Report[key].get("copied content", Report[key].get("copied_content", "")))
                    else:
                        if self.args.add_missing:
                            rewritten_content = Report[key].get("rewritten content", Report[key].get("rewritten_content", ""))
                            if rewritten_content != '':
                                cache_pseudo_response.append(rewritten_content)

                pseudo_response = self.tokenizer.batch_encode_plus(
                    cache_pseudo_response,
                    add_special_tokens=False,
                )
                pseudo_ids = pseudo_response['input_ids']
                pseudo_ids_tensor = torch.tensor([item for sublist in pseudo_ids for item in sublist], device=self.accelerator.device)
                pseudo_ids_tensor = torch.cat([pseudo_ids_tensor, torch.tensor([2], device=self.accelerator.device)])

                generated_response = self.tokenizer.batch_encode_plus(
                    cache_generated_response,
                    add_special_tokens=False,
                )
                generated_ids = generated_response['input_ids']
                generated_ids_tensor = torch.tensor([item for sublist in generated_ids for item in sublist], device=self.accelerator.device)
                generated_ids_tensor = torch.cat([generated_ids_tensor, torch.tensor([2], device=self.accelerator.device)])

                Pseudo_response.append(" ".join(cache_pseudo_response))
                Pseudo_response_ids.append(pseudo_ids_tensor)
                Generated_response.append(" ".join(cache_generated_response))
                Generated_response_ids.append(generated_ids_tensor)

                report_json.append(Report)
            except Exception as e:
                # ! If calling API false or the returned content is not consistent with our format:
                lg.warning(f"API Calling failed at step {i}! setting generated response as pseudo response")
                print(e)
                Pseudo_response.append(copy.deepcopy(text_responses[i]))
                Generated_response.append(copy.deepcopy(text_responses[i]))
                Pseudo_response_ids.append(copy.deepcopy(responses[i]))
                Generated_response_ids.append(copy.deepcopy(responses[i]))
                Pseudo_rewards.append(torch.zeros_like(responses[i], dtype=compute_type))
                Original_rewards.append(torch.zeros_like(responses[i], dtype=compute_type))
                report_json.append('')

        return {
            'Pseudo_response': Pseudo_response,
            'Pseudo_response_ids': Pseudo_response_ids,
            'Generated_response': Generated_response,
            'Generated_response_ids': Generated_response_ids,
            'report_json': report_json
        }


    @torch.inference_mode()
    def rollout(self, queries_data, step_idx) -> Dict[str, torch.Tensor]:
        self.policy.eval()
        static_messages = [{"role": "system", "content": PROMPT_LONG_coco_4V +
                f"input format: {json.dumps(input_format_coco_4V)} \n \n output format: {json.dumps(output_format_coco_4V)}"},]

        queries_all = []
        text_rsp = []
        text_standard_rsp = []
        text_pseudo_rsp = []
        text_generate_rsp = []
        report_json = []
        captions_all = []

        metadatas = []
        images_url_list = []
        images_bytes_list = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            total=len(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            gc.collect()
            torch.cuda.empty_cache()
            (
                images,
                images_path,
                images_url,
                images_bytes,
                queries,
                query_attn_masks,
                standard_responses,
            ) = common_utils.unpack_dict(
                common_utils.prepare_inputs(batch, device=self.accelerator.device),
                keys=(
                    "images",
                    "images_path",
                    "images_url",
                    "images_bytes",
                    "queries",
                    "query_attn_masks",
                    "standard_responses",
                ),
            )
            metadatas.append([path for path in images_path])
            images_url_list.append(images_url)
            images_bytes_list.append(images_bytes)
            if self.args.bf16:
                images = images.to(torch.bfloat16)
            elif self.args.fp16:
                images = images.half()

            compute_type = (
                torch.float16 if self.args.fp16 else (torch.bfloat16 if self.args.bf16 else torch.float32)
            )

            respond_outputs = self.policy.generate(
                inputs=queries,
                images=images,
                attention_mask=query_attn_masks,
                do_sample=True,
                max_new_tokens=self.args.response_len,
                pad_token_id=self.tokenizer.pad_token_id,
                suppress_tokens=(
                    [self.tokenizer.eos_token_id]
                    if self.args.suppress_eos_at_generation
                    else None
                ),
                top_p=self.args.top_p,
                top_k=self.args.top_k,
                temperature=self.args.temperature,
                num_return_sequences=1,
                synced_gpus=True if (torch.distributed.is_available() and torch.distributed.is_initialized()) else False,
            )
            responses = respond_outputs[:, queries.size(1):]

            additional_token1 = self.tokenizer.encode("?", add_special_tokens=False)[0]
            assert additional_token1 == 1577

            additional_token2 = self.tokenizer.encode("\n?")[-1]
            assert additional_token2 == 29973

            responses = truncate_after_eos_with_padding(
                responses,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                additional_tokens=[additional_token1, additional_token2],
            )

            text_responses = self.tokenizer.batch_decode(
                responses,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            text_rsp.append(text_responses)

            text_standard_responses = self.tokenizer.batch_decode(
                standard_responses,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            text_standard_rsp.append(text_standard_responses)

            reformate_queries = queries.clone()
            reformate_queries[reformate_queries==IMAGE_TOKEN_INDEX] = 1
            text_queries = self.tokenizer.batch_decode(
                reformate_queries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            text_queries = [query[query.find('USER:  \n')+8:query.find(' ASSISTANT:')] for query in text_queries]
            results = self.process_responses_coco_gpt4v(responses, images_url, text_queries, text_responses, text_standard_responses, static_messages, compute_type)
            queries_all.append(["<image>\n"+query for query in text_queries])

            Pseudo_response = results['Pseudo_response']
            Generated_response = results['Generated_response']
            report_json.extend(results['report_json'])

            text_pseudo_rsp.append(Pseudo_response)
            text_generate_rsp.append(Generated_response)

        metadatas = [item for sublist in metadatas for item in sublist]
        images_url_list = [item for sublist in images_url_list for item in sublist]
        images_bytes_list = [item for sublist in images_bytes_list for item in sublist]

        text_response_all = [item for sublist in text_rsp for item in sublist]
        text_standard_response_all = [item for sublist in text_standard_rsp for item in sublist]

        text_pseudo_response_all = [item for sublist in text_pseudo_rsp for item in sublist]
        text_generate_response_all = [item for sublist in text_generate_rsp for item in sublist]

        queries_all = [item for sublist in queries_all for item in sublist]
        response_dict = {
            "query": queries_all,
            "image_id": metadatas,
            "standard_response": text_standard_response_all,
            "original_generate_response": text_response_all,
            "AI_generate_response": text_generate_response_all,
            "AI_pseudo_response": text_pseudo_response_all,
            "AI_json_report": report_json,
            "image_bytes": images_bytes_list,
        }
        return response_dict

    def record_step_stats(self, step_idx, response_dict):
        if self.args.output_dir is not None:
            rollouts_to_disk = pd.DataFrame(response_dict).to_dict(
                orient="records"
            )
            if "image_bytes" in response_dict.keys():
                for item in rollouts_to_disk:
                    item["image_bytes"] = base64.b64encode(item["image_bytes"]).decode("utf-8")
            rollout_log_dir = os.path.join(self.args.output_dir, "rollouts")
            os.makedirs(rollout_log_dir, exist_ok=True)
            global_rank = int(os.environ.get("RANK", 0))
            with open(
                os.path.join(rollout_log_dir, f"step{step_idx}_rank{global_rank}.json"),
                "w",
            ) as f:
                json.dump(rollouts_to_disk, f, indent=4)
        return None
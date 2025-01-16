import abc
import logging
from typing import Dict, Optional

import torch
import transformers
from torch import Tensor, nn

from utils.common_utils import right_pad, compute_logprobs
from loguru import logger as lg

logger = logging.getLogger(__name__)


class Policy(nn.Module, abc.ABC):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.adapter_name = adapter_name

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        return self._post_respond(
            self._respond(
                queries,
                query_attn_masks,
                images,
                temperature,
                num_return_sequences,
            )
        )

    @abc.abstractmethod
    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _post_respond(self, respond_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return respond_outputs


class AutoregressivePolicy(Policy):
    def forward(
        self,
        images: Tensor,
        queries: Tensor,
        queries_attn_masks: Tensor,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        
        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
        self.base_model.config.use_cache = False

        if temperature is None:
            temperature = self.args.temperature

        keys = kwargs.keys()
        responses_key = [key for key in keys if "response" in key and "_mask" not in key and "scores" not in key and "image_relations" not in key]
        mask_key = [key+'_attention_mask' for key in responses_key]

        input_ids = []
        attention_masks_set = []
        images_set = []
        for key in responses_key:
            input_ids.append(torch.cat([queries, kwargs[key]], dim=1))
            if queries.size(1) == queries_attn_masks.size(1):
                attention_mask = input_ids[-1].ne(self.base_tokenizer.pad_token_id)
                attention_mask[:, : queries.size(1)] = queries_attn_masks
            else:
                attention_mask = kwargs[key].ne(self.base_tokenizer.pad_token_id)
                attention_mask = torch.cat([queries_attn_masks, attention_mask], dim=1)
            attention_masks_set.append(attention_mask)
            images_set.append(images)

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks_set, dim=0)
        images = torch.cat(images_set, dim=0)
        response_mask = ~input_ids[:, queries.size(1):].eq(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_masks,
            images=images,
            use_cache=False,
        )
        outputs = self.base_model(**inputs, output_hidden_states=True)
        original_logits = outputs.logits[:, -self.args.response_len - 1 : -1]
        logits = original_logits / temperature
        labels = input_ids[:, -self.args.response_len :]
        logprobs = compute_logprobs(
            logits, labels, ignore_index=self.base_tokenizer.pad_token_id
        )
        logprobs = logprobs * response_mask
        entropies = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
        last_hidden_state = outputs.hidden_states[-1][
            :, -self.args.response_len - 1 : -1
        ]
        entropies = entropies * response_mask


        total_bs = logprobs.size(0)//len(responses_key)
        return_dict = {}

        for i in range(len(responses_key)):
            return_dict.update({
                responses_key[i]+"_logprobs": logprobs[i*total_bs:(i+1)*total_bs],
                responses_key[i]+"_entropies": entropies[i*total_bs:(i+1)*total_bs],
            })

        return return_dict

    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
            # lg.info(self.adapter_name)
        self.base_model.config.use_cache = True
        self.base_model.config.cache_shape = (
            queries.shape[-1]
            + self.args.response_len
            + self.base_model.get_vision_tower().num_patches,
        )

        if temperature is None:
            temperature = self.args.temperature
        sequences = self.base_model.generate(
            inputs=queries,
            images=images,
            attention_mask=query_attn_masks,
            do_sample=True,
            max_new_tokens=self.args.response_len,
            pad_token_id=self.base_tokenizer.pad_token_id,
            suppress_tokens=(
                [self.base_tokenizer.eos_token_id]
                if self.args.suppress_eos_at_generation
                else None
            ),
            top_p=1.0,
            top_k=0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            synced_gpus=True if (torch.distributed.is_available() and torch.distributed.is_initialized()) else False,
        )
        responses = right_pad(
            sequences[:, queries.size(1) :],
            target_size=(sequences.size(0), self.args.response_len),
            value=self.base_tokenizer.pad_token_id,
        )
        return dict(
            responses=responses
        )  # Size (bsz * num_return_sequences, response_len).

def make_policy_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    adapter_name: Optional[str] = "default",
) -> Policy:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressivePolicy(
            args, base_model, base_tokenizer, adapter_name=adapter_name
        )

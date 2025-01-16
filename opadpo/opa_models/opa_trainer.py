import os

import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import has_length

from transformers.utils import is_peft_available
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import PeftModel

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

class LLaVATrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        entropies = -(outputs['logits'].softmax(dim=-1) * outputs['logits'].log_softmax(dim=-1)).sum(dim=-1)
        if self.args.entropy_loss:
            if self.args.entropy_mask_method == "random" or self.args.entropy_mask_method == "blockwise":
                new_images = torch.clone(inputs['images']).detach()
                masked_images = torch.stack([mask_single_image(new_images[i].unsqueeze(0),
                                                            self.args.entropy_mask_ratio,
                                                            self.args.entropy_mask_method)
                                            for i in range(new_images.size(0))]
                                            ).squeeze(1)
                inputs['images'] = masked_images
            elif self.args.entropy_mask_method == "attention":
                new_masks = torch.clone(inputs['attention_mask']).detach()
                image_attention_mask = new_masks.new_full((new_masks.size(0), 1369), True)
                image_attention_mask = mask_percentage_per_row(image_attention_mask, self.args.entropy_mask_ratio)
                # NOTE: the image mask must be catted before the response mask.
                # NOTE: the changes on the source code are in llava_arch.py (_get_multimodal_embeddings_for_sample)
                inputs['attention_mask'] = torch.cat([image_attention_mask, new_masks], dim=1)

            new_outputs = model(**inputs)
            new_entropies = -(new_outputs['logits'].softmax(dim=-1) * new_outputs['logits'].log_softmax(dim=-1)).sum(dim=-1)
            entropy_mask = (inputs['labels']!=-100)
            pad_size = new_entropies.size(-1) - entropy_mask.size(-1)
            entropy_mask = F.pad(entropy_mask, (pad_size, 0), value=False)
            entropy_loss = -((new_entropies - entropies) * entropy_mask).sum(-1) / entropy_mask.sum(dim=-1)
            entropy_loss = entropy_loss.mean()
            mask_sft_loss_item = new_outputs["loss"].item()
            entropy_loss_item = entropy_loss.item()
        else:
            mask_sft_loss_item = 0.0
            entropy_loss_item = 0.0

        self.log({"base_sft_loss": outputs["loss"].item(),
                    "mask_sft_loss": mask_sft_loss_item,
                    "entropy_loss": entropy_loss_item,})

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if is_peft_available() and isinstance(model, PeftModel):
                model_name = unwrap_model(model.base_model)._get_name()
            else:
                model_name = unwrap_model(model)._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            if self.args.entropy_loss:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                self.args.entropy_loss_coef *= self.args.entropy_decay_coef
                loss += entropy_loss * self.args.entropy_loss_coef
            else:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

def mask_percentage(matrix: torch.Tensor, percentage: float) -> torch.Tensor:
    num_elements = matrix.numel()
    num_to_mask = int(num_elements * percentage)
    indices = torch.randperm(num_elements)[:num_to_mask]
    flat_matrix = matrix.view(-1)
    flat_matrix[indices] = False
    return flat_matrix.view(matrix.size())

def mask_percentage_per_row(matrix: torch.Tensor, percentage: float) -> torch.Tensor:
    num_columns = matrix.size(1)
    num_to_mask_per_row = int(num_columns * percentage)
    for i in range(matrix.size(0)):
        indices = torch.randperm(num_columns)[:num_to_mask_per_row]
        matrix[i, indices] = False
    return matrix

def mask_single_image(image, mask_percentage, mask_method='random'):
    mean_value = image.mean()
    _, C, H, W = image.shape
    if mask_method == 'random':
        total_pixels = H * W
        mask_pixels = int(total_pixels * mask_percentage)
        mask_indices = torch.randperm(total_pixels)[:mask_pixels]
        flat_image = image.view(C, -1)
        flat_image[:, mask_indices] = mean_value
    elif mask_method == 'blockwise':
        block_size = 14
        H_blocks = H // block_size
        W_blocks = W // block_size
        total_blocks = H_blocks * W_blocks
        mask_blocks = int(total_blocks * mask_percentage)
        mask_indices = torch.randperm(total_blocks)[:mask_blocks]

        flat_image = image.view(C, H_blocks, block_size, W_blocks, block_size)
        for idx in mask_indices:
            h = idx // W_blocks
            w = idx % W_blocks
            flat_image[:, h, :, w, :] = mean_value
    else:
        raise NotImplementedError
    masked_image = flat_image.view(1, C, H, W)
    return masked_image

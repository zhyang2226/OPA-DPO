import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.model import *
from PIL import Image
import math
from peft import PeftModel

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    compute_dtype = torch.float16
    if args.use_qlora:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        bits = 16
        dtype = torch.bfloat16
        compute_dtype = torch.bfloat16

        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            device_map={"": "cuda:0"},
            torch_dtype=dtype,
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["mm_projector", "lm_head"],
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        model = PeftModel.from_pretrained(
            model,
            args.qlora_path,
        )

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=compute_dtype)
        image_processor = vision_tower.image_processor
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, args.model_base, model_name
        )

    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    ans_scores = { 'yes': [], 'no': [], 'Yes': [], 'No': [], '\nYes': [], '\nNo': [] }
    label_id = {    
            'yes': tokenizer(" yes", return_tensors="pt")['input_ids'][0][-1].item(),  
            'no': tokenizer(" no", return_tensors="pt")['input_ids'][0][-1].item(),
            'Yes': tokenizer(" Yes", return_tensors="pt")['input_ids'][0][-1].item(),
            'No': tokenizer(" No", return_tensors="pt")['input_ids'][0][-1].item(),
            '\nYes': tokenizer("\nYes", return_tensors="pt")['input_ids'][0][-1].item(),
            '\nNo': tokenizer("\nNo", return_tensors="pt")['input_ids'][0][-1].item() }
    if args.logit_bias != 0:
        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, use_fast=False, add_prefix_space=True)
        def get_tokens_as_tuple(word):
            return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])
        sequence_bias = {get_tokens_as_tuple("Yes"): -args.logit_bias, get_tokens_as_tuple("No"): args.logit_bias}
        # print(sequence_bias)
        # exit()
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        # image_file = 'COCO_val2014_' + image_file
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        if args.short_eval:
            conv.system += " Please describe the image and then answer the question."
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(prompt)
        # exit()
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )

        image = Image.open(os.path.join(args.image_folder, image_file))
        if args.image_aspect_ratio == 'pad':
            image = image.convert('RGB')
            def expand2square(pil_img, background_color):
                # print(background_color)
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
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        model.config.use_cache = True
        model.config.cache_shape = (2048,)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=compute_dtype).cuda(),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=64 if args.short_eval else 1024,
                # stopping_criteria=[stopping_criteria],
                use_cache=True,
                return_dict_in_generate=args.output_scores,
                output_scores=args.output_scores,
                sequence_bias=sequence_bias if args.logit_bias != 0 else None,
            )
            if args.output_scores:
                for it in ans_scores:
                    ans_scores[it].append(output_ids.scores[0][0][label_id[it]].item())     
                # print(ans_scores)       
                continue
                print(output_ids.sequences.shape, input_ids.shape[1], len(output_ids.scores))
                print(output_ids.scores[0].shape)
                print(ans_scores)
                print('yes', tokenizer("yes", return_tensors="pt")['input_ids'][0][1]) #4874
                print('no', tokenizer("no", return_tensors="pt")['input_ids'][0][1]) # 694
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        # exit()
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        ans_file.flush()
    if args.output_scores:
        ans_file.write(json.dumps(ans_scores))
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-qlora", type=bool, default=False)
    parser.add_argument("--qlora-path", type=str, default="")
    parser.add_argument("--short_eval", type=bool, default=False)
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')
    parser.add_argument("--output_scores", type=bool, default=False)
    parser.add_argument("--logit-bias", type=float, default=0)
    args = parser.parse_args()

    eval_model(args)

# ! This script is used to evaluate the model on multiple benchmarks.
# ! Evaluation requires only one GPU, but you should at least guarantee that the GPU memory larger than 40 GB to prevent OOM problem.
export CUDA_VISIBLE_DEVICES=3
export OPENAI_ENDPOINT="YOUR_ENDPOINT"
export OPENAI_API_KEY='YOUR_API_KEY'
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "PYTHONPATH=$PYTHONPATH"

# # * FOR LLaVA-1.5-7B Model OPA Evaluation
MODEL_BASE=./base_models/llava-v1.5-7b
MODEL_LORA_BASE=./output/llava7b_opa_model/checkpoint-final/adapter_model/lora_policy/
MODEL_SUFFIX=llava7b_opa_eval

# # * FOR LLaVA-1.5-7B Model OPA-DPO Evaluation
# MODEL_BASE=./base_models/llava-v1.5-7b
# MODEL_LORA_BASE=./output/llava7b_opadpo_model/checkpoint-final/adapter_model/lora_policy/
# MODEL_SUFFIX=llava7b_opadpo_eval

# # * FOR LLaVA-1.5-13B Model OPA Evaluation
# MODEL_BASE=./base_models/llava-v1.5-13b
# MODEL_LORA_BASE=./output/llava13b_opa_model/checkpoint-final/adapter_model/lora_policy/
# MODEL_SUFFIX=llava13b_opa_eval

# # * FOR LLaVA-1.5-13B Model OPA-DPO Evaluation
# MODEL_BASE=./base_models/llava-v1.5-13b
# MODEL_LORA_BASE=./output/llava13b_opadpo_model/checkpoint-final/adapter_model/lora_policy/
# MODEL_SUFFIX=llava13b_opadpo_eval

# ! Define important Figure Path
IMAGE_FOLDER_LB=PATH_TO/coco/train2017
IMAGE_FOLDER_POPE=PATH_TO/coco/val2014
IMAGE_DIR_AMBER=PATH_TO/AMBER/image
ANNOTATION_FILE=PATH_TO/coco/annotations

# ! Stage1 Eval mm-hal bench (requires OpenAI API)
OUTPUT_DIR=./output/evaluation/mmhal_bench

python ./eval_llava_rlhf_coco/model_vqa_mmhal.py \
    --model-path ${MODEL_BASE} \
    --use-qlora True \
    --qlora-path ${MODEL_LORA_BASE} \
    --temperature 0.0 \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --image_aspect_ratio pad \
    --test-prompt ''

python ./eval_llava_rlhf_coco/eval_gpt_mmhal.py \
    --response ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --evaluation ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}.jsonl \
    --gpt-model gpt-4

python ./eval_llava_rlhf_coco/summarize_gpt_mmhal.py \
    --evaluation ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}.jsonl

# ! Stage2 Eval LLaVA bench (requires OpenAI API)
OUTPUT_DIR=./output/evaluation/llava_bench

python ./eval_llava_rlhf_coco/model_vqa.py \
    --model-path ${MODEL_BASE} \
    --use-qlora True \
    --qlora-path ${MODEL_LORA_BASE} \
    --question-file ./eval_llava_rlhf_coco/llava/qa90_questions.jsonl \
    --image-folder ${IMAGE_FOLDER_LB} \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --image_aspect_ratio pad \
    --test-prompt ''

python ./eval_llava_rlhf_coco/eval_gpt_review_visual.py \
    --question ./eval_llava_rlhf_coco/llava/qa90_questions.jsonl \
    --context ./eval_llava_rlhf_coco/table/caps_boxes_coco2014_val_80.jsonl \
    --answer-list \
    ./eval_llava_rlhf_coco/llava/qa90_gpt4_answer.jsonl \
    ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --rule ./eval_llava_rlhf_coco/table/rule.json \
    --output ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}.jsonl

python ./eval_llava_rlhf_coco/summarize_gpt_review.py -d ${OUTPUT_DIR}/review/ -f review-file-${MODEL_SUFFIX}.jsonl

# ! Stage3 Eval POPE bench (No need for OpenAI API)
OUTPUT_DIR=./output/evaluation/pop_bench

POPE_CAT="adversarial"
echo ${MODEL_SUFFIX} ${POPE_CAT}
python ./eval_llava_rlhf_coco/model_vqa.py \
    --short_eval True \
    --model-path ${MODEL_BASE} \
    --use-qlora True \
    --qlora-path ${MODEL_LORA_BASE} \
    --question-file ./eval_llava_rlhf_coco/pope/coco_pope_${POPE_CAT}.jsonl \
    --image-folder ${IMAGE_FOLDER_POPE} \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}_${POPE_CAT}.jsonl \
    --image_aspect_ratio pad \
    --test-prompt '\nAnswer the question using a single word or phrase.'

python ./eval_llava_rlhf_coco/summarize_eval_pope.py \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}_${POPE_CAT}.jsonl \
    --label-file ./eval_llava_rlhf_coco/pope/coco_pope_${POPE_CAT}.jsonl

# ! Stage4 Eval AMBER bench (No need for OpenAI API)
OUTPUT_DIR=./output/evaluation/AMBER_bench

python -m spacy download en_core_web_lg

python ./eval_llava_rlhf_coco/AMBER_generate.py \
    --model-path ${MODEL_BASE} \
    --use-qlora True \
    --qlora-path ${MODEL_LORA_BASE} \
    --temperature 0.0 \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --image-file $IMAGE_DIR_AMBER \
    --image_aspect_ratio pad \
    --test-prompt ''

python ./eval_llava_rlhf_coco/AMBER_eval.py \
    --inference_data ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --evaluation_type 'g'


# ! Stage5 Eval Obj-Hal bench (requires OpenAI API, but is optional)
OUTPUT_DIR=./output/evaluation/objhal_bench

python ./eval_llava_rlhf_coco/model_vqa_objectHal.py \
    --model-path ${MODEL_BASE} \
    --use-qlora True \
    --qlora-path ${MODEL_LORA_BASE} \
    --question-file ./eval_llava_rlhf_coco/object_hal/obj_halbench_300_with_image.jsonl \
    --answers-file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --image_aspect_ratio pad \
    --test-prompt ''

pip install jsonlines
python -m spacy download en_core_web_trf

# NOTE: USE_GPT (IF DO_NOT_USE_GPT, do not set --use_gpt)
python ./eval_llava_rlhf_coco/eval_gpt_obj_halbench.py \
    --coco_path ${ANNOTATION_FILE} \
    --answers_file ${OUTPUT_DIR}/answer/answer-file-${MODEL_SUFFIX}.jsonl \
    --evaluation_file ${OUTPUT_DIR}/review/review-file-${MODEL_SUFFIX}_eval.jsonl \
    --gpt-model gpt-4 \
    --use_gpt 
    

# there is some version conflict in the package, en_core_web_trf requires tokenizers-0.13.3 transformers-4.30.2.
# We need transformers-4.34.1 and tokenizers-0.14.1 for the model. Thereby we install back the two packages after the evaluation.
pip install transformers==4.34.1
pip install tokenizers==0.14.1
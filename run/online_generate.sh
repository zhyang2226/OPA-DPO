#!/bin/bash

set -e
set -x

# FOR MultiCard RUN
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$PWD:$PYTHONPATH"

# Note: For LLaVA-1.5-7B Model Generation (Subset1)
export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset1"
export MODEL_DIR="./base_models/llava-v1.5-7b"
export OUTPUT_DIR="./output/llava7b_online_generation_subset1"
export POLICY_LORA_DIR="none"
# # Note: For LLaVA-1.5-7B Model Generation (Subset2)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset2"
# export MODEL_DIR="./base_models/llava-v1.5-7b"
# export OUTPUT_DIR="./output/llava7b_online_generation_subset2"
# export POLICY_LORA_DIR="none"
# # Note: For LLaVA-1.5-7B Model Generation (Subset3)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset3"
# export MODEL_DIR="./base_models/llava-v1.5-7b"
# export OUTPUT_DIR="./output/llava7b_online_generation_subset3"
# export POLICY_LORA_DIR="none"
# # Note: For LLaVA-1.5-7B Model Generation (Subset4)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset4"
# export MODEL_DIR="./base_models/llava-v1.5-7b"
# export OUTPUT_DIR="./output/llava7b_online_generation_subset4"
# export POLICY_LORA_DIR="none"

# # Note: For LLaVA-1.5-13B Model Generation (Subset1)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset1"
# export MODEL_DIR="./base_models/llava-v1.5-13b"
# export OUTPUT_DIR="./output/llava13b_online_generation_subset1"
# export POLICY_LORA_DIR="none"
# # Note: For LLaVA-1.5-13B Model Generation (Subset2)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset2"
# export MODEL_DIR="./base_models/llava-v1.5-13b"
# export OUTPUT_DIR="./output/llava13b_online_generation_subset2"
# export POLICY_LORA_DIR="none"
# # Note: For LLaVA-1.5-13B Model Generation (Subset3)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset3"
# export MODEL_DIR="./base_models/llava-v1.5-13b"
# export OUTPUT_DIR="./output/llava13b_online_generation_subset3"
# export POLICY_LORA_DIR="none"
# # Note: For LLaVA-1.5-13B Model Generation (Subset4)
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset4"
# export MODEL_DIR="./base_models/llava-v1.5-13b"
# export OUTPUT_DIR="./output/llava13b_online_generation_subset4"
# export POLICY_LORA_DIR="none"

# # Note: For Single Card Local testing
# export PYTHONPATH="$PWD:$PYTHONPATH"
# export CUDA_VISIBLE_DEVICES=0
# export GPUS_PER_NODE=1
# export DATA_DIR="./base_datasets/LLaVA-RLAIF-SubData/subset1"
# export MODEL_DIR="./base_models/llava-v1.5-7b"
# export POLICY_LORA_DIR="none"
# export OUTPUT_DIR="./output/llava7b_online_generation_subset1"

# Note: For Azure API. If You have multiple endpoints, you can adjust API_NUM, and add more API_KEY, AZURE_POINT, and API_MODEL.
export API_NUM=1
export API_KEY1='YOUR API KEY'            # TODO: Change this to your API key
export AZURE_POINT1="YOUR AZURE ENDPOINT" # TODO: Change this to your Azure endpoint
export API_MODEL1="gpt-4_vision-preview"  # TODO: Maybe you need to modify this to proper version of GPT-4V


# TRAINING CONFIG
EPOCH=1
SEED=42
TEMPERATURE=1.0
TOPK=30
TOPP=0.95
PHASE=0
SAMPLE=2500

# For 40GB-A100
ROLLOUT_BATCH_SIZE=32
STEP_BATCH_SIZE=32
ROLLOUT_PER_DEVICE_BATCH_SIZE=4
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=4
STEP_PER_DEVICE_BATCH_SIZE=4
NOPTEPOCHS=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    ./opadpo/online_generation_custom.py \
    --cfg 'configs/llava/llava_online_generation.yaml' \
    --local-rank 0 \
    --base_model_name $MODEL_DIR \
    --base_model $MODEL_DIR \
    --policy_model_name_or_path $POLICY_LORA_DIR \
    --output_dir $OUTPUT_DIR \
    --image_folder $DATA_DIR \
    --data_path $DATA_DIR \
    --seed $SEED \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --step_batch_size $STEP_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --total_epochs $EPOCH \
    --model_max_length 2048 \
    --query_len 128 \
    --response_len 896 \
    --noptepochs $NOPTEPOCHS \
    --mm_vision_select_layer -2 \
    --ddp_backend 'nccl' \
    --top_k $TOPK \
    --top_p $TOPP \
    --phase $PHASE \
    --sample_num $SAMPLE
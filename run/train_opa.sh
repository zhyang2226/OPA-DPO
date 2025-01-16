#!/bin/bash
set -e
set -x

# Note: For LLaVA-1.5-7B Model OPA Training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4
export DATA_DIR="./base_datasets/opa_training_data-7B"
export IMAGE_DIR="none"
export MODEL_DIR="./base_models/llava-v1.5-7b"
export OUTPUT_DIR="./output/llava7b_opa_model"

# # Note: For LLaVA-1.5-13B Model OPA Training
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export GPUS_PER_NODE=4
# export DATA_DIR="./base_datasets/opa_training_data-13B"
# export IMAGE_DIR="none"
# export MODEL_DIR="./base_models/llava-v1.5-13b"
# export OUTPUT_DIR="./output/llava13b_opa_model"

# # ! FOR Local RUN
# export CUDA_VISIBLE_DEVICES=0
# export GPUS_PER_NODE=1
# export DATA_DIR="./base_datasets/opa_training_data-7B"
# export IMAGE_DIR="none"
# export MODEL_DIR="./base_models/llava-v1.5-7b"
# export OUTPUT_DIR="./output/llava7b_opa_model"

export PYTHONPATH="$PWD:$PYTHONPATH"

PERDEVICE_BS=4
GRADIENT_ACC=8
EPOCH=2
DEEPSPEED='opadpo/deepspeed_stage_1_config.json'

ENTROPY_LOSS="False"
ENTROPY_MASK_RAIO=0.8
ENTROPY_MASK_METHOD='random'
ENTROPY_LOSS_COEF=0.01
ENTROPY_DECAY_COEF=1.0

LORATUNE='True'
LORA_RANK=256
LORA_ALPHA=512
LORA_DROP=0.0

FULLTUNE='False'
TUNE_MM_PROJECT='True'
TUNE_BASE_MODEL='True'
TUNE_VISION_TOWER='True'

# Debugging: Print variables to check their values
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "DATA_DIR=$DATA_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"

if [ "$FULLTUNE" = "True" -a "$LORATUNE" = "True" ]; then
    echo "Error: FULLTUNE and LORATUNE cannot both be True"
    exit 1
fi
if [ "$FULLTUNE" = "False" -a "$LORATUNE" = "False" ]; then
    echo "Error: FULLTUNE and LORATUNE cannot both be False"
    exit 1
fi

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    ./opadpo/opa_train_custom.py \
    --cfg 'configs/llava/llava_opa.yaml' \
    --base_model $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --image_folder $IMAGE_DIR \
    --data_dir $DATA_DIR \
    --per_device_train_batch_size $PERDEVICE_BS \
    --per_device_eval_batch_size $PERDEVICE_BS \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --deepspeed $DEEPSPEED \
    --tf32 \
    --bf16 \
    --use_flash_attention \
    --save_steps 40 \
    --eval_steps 10 \
    --mm_vision_select_layer -2 \
    --mm_projector_type "mlp2x_gelu" \
    --full_tune $FULLTUNE \
    --tune_mm_mlp_adapter $TUNE_MM_PROJECT \
    --tune_base_model $TUNE_BASE_MODEL \
    --tune_vision_tower $TUNE_VISION_TOWER \
    --lora_tune $LORATUNE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_drop $LORA_DROP \
    --num_train_epochs $EPOCH \
    --entropy_loss $ENTROPY_LOSS \
    --entropy_mask_ratio $ENTROPY_MASK_RAIO \
    --entropy_mask_method $ENTROPY_MASK_METHOD \
    --entropy_loss_coef $ENTROPY_LOSS_COEF \
    --entropy_decay_coef $ENTROPY_DECAY_COEF

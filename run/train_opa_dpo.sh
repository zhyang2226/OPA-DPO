#!/bin/bash
set -e
set -x

# * FOR Multicard RUN
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4

# Note: For LLaVA-1.5-7B Model OPA-DPO Training
export DATA_DIR="./base_datasets/opadpo_training_data-7B"
export IMAGE_DIR="none"
export MODEL_DIR="./base_models/llava-v1.5-7b"
export POLICY_LORA_DIR="./output/llava7b_opa_model/checkpoint-final"
export OUTPUT_DIR="./output/llava7b_opadpo_model"

# # Note: For LLaVA-1.5-13B Model OPA-DPO Training
# export DATA_DIR="./base_datasets/opadpo_training_data-13B"
# export IMAGE_DIR="none"
# export MODEL_DIR="./base_models/llava-v1.5-13b"
# export POLICY_LORA_DIR="./output/llava13b_opa_model/checkpoint-final"
# export OUTPUT_DIR="./output/llava13b_opadpo_model"

# # ! FOR Local test
# export CUDA_VISIBLE_DEVICES=0
# export GPUS_PER_NODE=1
# export DATA_DIR="./base_datasets/opadpo_training_data-7B"
# export IMAGE_DIR="none"
# export MODEL_DIR="./base_models/llava-v1.5-7b"
# export POLICY_LORA_DIR="./output/llava7b_opa_model/checkpoint-final"
# export OUTPUT_DIR="./output/llava7b_opadpo_model"


export PYTHONPATH="$PWD:$PYTHONPATH"

LORA_RANK=256
LORA_ALPHA=512
LORA_DROP=0.0

ROLLOUT_BATCH_SIZE=64
STEP_BATCH_SIZE=32
ROLLOUT_PER_DEVICE_BATCH_SIZE=2
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=2
STEP_PER_DEVICE_BATCH_SIZE=2
NOPTEPOCHS=1

# TRAINING CONFIG
LEARNING_RATE=1e-6
EPOCH=4
WARMUP_STEPS=5
MAX_STEP=500

LR_DECAY='cosine'
GRAD_NORM=1.0
SEED=42
TRAIN_FROM_SFT='True'

TEMPERATURE=1.0
REFEERENCE_FREE="False"
F_DIV_TYPE="reverse_kl"
LOSS_TYPE="sigmoid"
BETA=0.1
LABEL_SMOOTH=0.0

COPO='True'
COPO_RATIO=0.3
COPO_METHOD='random'
COPO_COEF=0.2
ANCPO='True'
ANCHOR=0.0
MDPO_ANCHOR='True'
ANCHOR_COEF=1.0

DETAILED_REPORT='True'
RESPONSE_SCORE='True'
RESPONSE_IMAGE_RELATION='True'
STANDARD_PAIR_COEF=1.0
AI_PAIR_COEF=1.0

# Debugging: Print variables to check their values
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "DATA_DIR=$DATA_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "POLICY_LORA_DIR=$POLICY_LORA_DIR"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LEARNING_RATE=$LEARNING_RATE"
echo "KL_COEF=$KL_COEF"
echo "EPOCH=$EPOCH"
echo "WARMUP_STEPS=$WARMUP_STEPS"
echo "ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE"
echo "STEP_BATCH_SIZE=$STEP_BATCH_SIZE"
echo "ROLLOUT_PER_DEVICE_BATCH_SIZE=$ROLLOUT_PER_DEVICE_BATCH_SIZE"
echo "REWARD_MODEL_PER_DEVICE_BATCH_SIZE=$REWARD_MODEL_PER_DEVICE_BATCH_SIZE"
echo "STEP_PER_DEVICE_BATCH_SIZE=$STEP_PER_DEVICE_BATCH_SIZE"
echo "NOPTEPOCHS=$NOPTEPOCHS"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    ./opadpo/opadpo_train_custom.py \
    --cfg 'configs/llava/llava_dpo.yaml' \
    --local-rank 0 \
    --base_model_name $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --image_folder $IMAGE_DIR \
    --data_path $DATA_DIR \
    --policy_model_name_or_path $POLICY_LORA_DIR \
    --seed $SEED \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --step_batch_size $STEP_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --total_epochs $EPOCH \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_drop $LORA_DROP \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --weight_decay 0.0 \
    --lr_scheduler_type $LR_DECAY \
    --logging_steps 1 \
    --whitening_async_stats 'full_batch' \
    --model_max_length 2048 \
    --query_len 128 \
    --response_len 896 \
    --noptepochs $NOPTEPOCHS \
    --eval_steps 5000 \
    --save_steps 10 \
    --save_total_limit 5 \
    --image_aspect_ratio 'pad' \
    --train_splits 'train' \
    --vision_tower 'different' \
    --mm_vision_select_layer -1 \
    --ddp_backend 'nccl' \
    --report_to 'wandb' \
    --norm_maintain_32 'False' \
    --value_head_mode 'linear' \
    --max_grad_norm $GRAD_NORM \
    --temperature $TEMPERATURE \
    --max_step $MAX_STEP \
    --train_from_sft $TRAIN_FROM_SFT \
    --reference_free $REFEERENCE_FREE \
    --f_divergence_type $F_DIV_TYPE \
    --loss_type $LOSS_TYPE \
    --beta $BETA \
    --label_smoothing $LABEL_SMOOTH \
    --CoPO $COPO \
    --CoPO_mask_ratio $COPO_RATIO \
    --CoPO_method $COPO_METHOD \
    --CoPO_coef $COPO_COEF \
    --AncPO $ANCPO \
    --Anchor_value $ANCHOR \
    --mDPO_anchor $MDPO_ANCHOR \
    --Anchor_coef $ANCHOR_COEF \
    --detailed_report $DETAILED_REPORT \
    --response_score $RESPONSE_SCORE \
    --response_image_relation $RESPONSE_IMAGE_RELATION \
    --standard_pair_coef $STANDARD_PAIR_COEF \
    --AI_pair_coef $AI_PAIR_COEF

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# first cli argument is the model size, default is 7
# second cli argument is the batch size per gpu, default is 1
MODEL_SIZE_ARG=${1:-7}
BATCH_SIZE_PER_GPU=${2:-1}

MODEL_SIZE="${MODEL_SIZE_ARG}b"
NUM_GPUS=8
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

MODEL_NAME="meta-llama/Llama-2-${MODEL_SIZE}-hf"
TOKENIZER_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="yentinglin/zh_TW_c4"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME},${DATASET_NAME},${MODEL_SIZE},stage3,no-offload"

echo "Training llama2 model: ${MODEL_NAME}"
echo "using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --use_flash_attn \
    --dataset_name $DATASET_NAME \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --output_dir output/zh_TW_c4_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to wandb \
    --logging_steps 1 \
    --checkpointing_steps 10000

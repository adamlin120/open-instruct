BATCH_SIZE_PER_GPU=1
NUM_GPUS=8
GRADIENT_ACC_STEPS=1

MODEL_NAME="yentinglin/zh_llama2_13b"
MODEL_NAME_STEM=${MODEL_NAME##*/}
TOKENIZER_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="yentinglin/zh_instruction"

export WANDB_PROJECT="Chinese-LLAMA2"
export WANDB_TAGS="${MODEL_NAME_STEM},${DATASET_NAME},instruction-tuning,stage3,no-offload"

export WANDB_API_KEY="94f8e06129c90551b50bfc5556e389fc574c2fa3"
export HUGGING_FACE_HUB_TOKEN="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"

echo "Training llama2 model: ${MODEL_NAME}"
echo "using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    open_instruct/finetune_trainer.py \
    --model_name_or_path "${MODEL_NAME}" \
    --tokenizer_name "${TOKENIZER_NAME}" \
    --use_fast_tokenizer True \
    --dataset_name "${DATASET_NAME}" \
    --max_seq_length 4096 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --output_dir output/sft/0729/${MODEL_NAME_STEM}/ \
    --bf16 True \
    --tf32 True \
    --overwrite_output_dir \
    --report_to "all" \
    --preprocessing_num_workers 8 \
    --hub_strategy "all_checkpoints" \
    --hub_model_id "${MODEL_NAME_STEM}-sft" \
    --hub_private_repo True \
    --push_to_hub True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --torch_dtype "bfloat16"

